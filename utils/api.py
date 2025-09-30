import os
import time
import logging
import json
import requests
import random
import string
import hashlib
from typing import Optional, Dict, Any, List # Added List
from dotenv import load_dotenv

# Import CORE engine for caching integration
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from core_gemini import Engine, EngineConfig
    from core_gemini.persistence.cache_index import CacheIndex
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    logging.warning("CORE engine not available, falling back to direct API calls")

# Import admin override system
try:
    from .admin_override import AdminOverrideManager, BatchManager
    ADMIN_OVERRIDE_AVAILABLE = True
except ImportError:
    ADMIN_OVERRIDE_AVAILABLE = False
    logging.warning("Admin override system not available")

load_dotenv()

class APIClient:
    """
    Client for interacting with LLM API endpoints (OpenAI or other).
    Supports 'test' and 'judge' configurations.
    """

    def __init__(self, model_type=None, request_timeout=240, max_retries=3, retry_delay=5):
        self.model_type = model_type or "default"

        # Initialize CORE engine if available
        self.use_core = CORE_AVAILABLE and os.getenv("USE_CORE_ENGINE", "true").lower() == "true"
        if self.use_core:
            # Configure CORE engine
            cfg = EngineConfig(
                backend="genai",  # Use native Google GenAI
                api_key=os.getenv("GEMINI_API_KEY"),
                use_vertex=os.getenv("USE_VERTEX", "false").lower() == "true",
                project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
                auto_cache_enabled=True,
                auto_cache_token_threshold=6000,  # Cache large contexts
                max_concurrency=16  # Reasonable concurrency for EQBench3
            )
            self.core_engine = Engine(cfg)
            self.cache_index = CacheIndex("logs/eqbench3_cache.sqlite")
            self.payload_cache = {}  # In-memory cache for payload deduplication

            # Initialize admin override system
            if ADMIN_OVERRIDE_AVAILABLE:
                self.admin_manager = AdminOverrideManager()
                self.batch_manager = BatchManager()
                logging.info("Admin override system enabled for cost control")
            else:
                self.admin_manager = None
                self.batch_manager = None
                logging.warning("Admin override system not available")

            logging.info(f"Initialized {self.model_type} API client with CORE engine")
        else:
            # Fallback to direct API calls
            # Load specific or default API credentials based on model_type
            if model_type == "test":
                self.api_key = os.getenv("TEST_API_KEY", os.getenv("OPENAI_API_KEY"))
                self.base_url = os.getenv("TEST_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
            elif model_type == "judge":
                # Judge model is used for ELO pairwise comparisons
                self.api_key = os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY"))
                self.base_url = os.getenv("JUDGE_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
            else: # Default/fallback
                self.api_key = os.getenv("OPENAI_API_KEY")
                self.base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

            self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", request_timeout))
            self.max_retries = int(os.getenv("MAX_RETRIES", max_retries))
            self.retry_delay = int(os.getenv("RETRY_DELAY", retry_delay))

            if not self.api_key:
                logging.warning(f"API Key for model_type '{self.model_type}' not found in environment variables.")
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            logging.debug(f"Initialized {self.model_type} API client with URL: {self.base_url}")

    def generate(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4000, min_p: Optional[float] = 0.1) -> str:
        """
        Generic chat-completion style call using a list of messages.
        Uses CORE engine with caching when available to avoid repeating identical payloads.
        Falls back to direct API calls if CORE is not available.
        """
        # Create cache key from model and messages to avoid repeating identical requests
        cache_key = hashlib.sha256(
            json.dumps({"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}, sort_keys=True).encode()
        ).hexdigest()

        # Check in-memory cache first
        if cache_key in self.payload_cache:
            logging.debug(f"Cache hit for payload {cache_key[:8]}...")
            return self.payload_cache[cache_key]

        if self.use_core:
            # Use CORE engine with caching
            try:
                # Convert model name for CORE engine (handle different naming conventions)
                core_model = self._convert_model_name(model)

                # Admin override check for expensive operations
                estimated_tokens = self._estimate_payload_tokens(messages, core_model)
                if (self.admin_manager and
                    not self.admin_manager.check_override_for_large_payload(core_model, estimated_tokens, f"eqbench3_{self.model_type}")):
                    logging.warning(f"Admin override denied for {core_model} with {estimated_tokens} estimated tokens")
                    # Fall through to OpenAI fallback instead
                    raise Exception("Admin override required - falling back to OpenAI")

                # Log the operation for audit
                if self.admin_manager:
                    self.admin_manager.log_operation(core_model, estimated_tokens, f"eqbench3_{self.model_type}", True)

                # Call CORE engine
                result = self.core_engine.acomplete(
                    messages=messages,
                    model=core_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    metadata={"eqbench3_model_type": self.model_type}
                )

                content = result["text"]

                # Strip thinking/reasoning blocks if present
                content = self._clean_response(content)

                # Cache the result
                self.payload_cache[cache_key] = content
                logging.debug(f"CORE engine call completed for {model}, cached as {cache_key[:8]}")

                return content

            except Exception as e:
                logging.error(f"CORE engine call failed for {model}: {e}")
                # Fall through to direct API call as backup

        # Fallback to direct API calls (original logic)
        if not hasattr(self, 'api_key') or not self.api_key:
            raise ValueError(f"Cannot make API call for '{self.model_type}'. API Key is missing.")

        for attempt in range(self.max_retries):
            response = None
            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

                # Apply min_p only for the test model if provided
                if self.model_type == "test" and min_p is not None:
                    payload['min_p'] = min_p
                    logging.debug(f"Applying min_p={min_p} for test model call.")
                elif self.model_type == "judge":
                    pass  # Don't add min_p for judge

                # Handle different API endpoints and model-specific requirements
                self._adjust_payload_for_endpoint(payload, model)

                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                data = response.json()

                if not data.get("choices") or not data["choices"][0].get("message") or "content" not in data["choices"][0]["message"]:
                    logging.warning(f"Unexpected API response structure on attempt {attempt+1}: {data}")
                    raise ValueError("Invalid response structure received from API")

                content = data["choices"][0]["message"]["content"]
                content = self._clean_response(content)

                # Cache the result
                self.payload_cache[cache_key] = content
                logging.debug(f"Direct API call completed for {model}, cached as {cache_key[:8]}")

                return content

            except requests.exceptions.Timeout:
                logging.warning(f"Request timed out on attempt {attempt+1}/{self.max_retries} for model {model}")
            except requests.exceptions.RequestException as e:
                self._handle_request_error(e, response, attempt, model)
            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON response on attempt {attempt+1}/{self.max_retries} for model {model}.")
                if response is not None:
                    logging.error(f"Raw response text: {response.text}")
            except Exception as e:
                logging.error(f"Unexpected error during API call attempt {attempt+1}/{self.max_retries} for model {model}: {e}", exc_info=True)

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"Failed to generate text for model {model} after {self.max_retries} attempts")

    def _convert_model_name(self, model: str) -> str:
        """Convert model names between different APIs."""
        # Handle common model name mappings
        if model.startswith("google/"):
            return model.replace("google/", "")
        elif model.startswith("openai/"):
            return model.replace("openai/", "")
        # Add more mappings as needed
        return model

    def _estimate_payload_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        """Estimate token count for payload before making API call."""
        if hasattr(self, 'core_engine'):
            try:
                # Use CORE engine's estimation for consistency
                from core_gemini.caching import openai_messages_to_rest_contents, estimate_tokens
                contents = openai_messages_to_rest_contents(messages)
                estimated = estimate_tokens(self.core_engine.cfg, model=model, contents=contents)
                return estimated or 1000  # fallback estimate
            except Exception:
                pass

        # Fallback: rough character-based estimation (very approximate)
        total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
        return total_chars // 3  # Very rough approximation

    def _clean_response(self, content: str) -> str:
        """Clean response content by stripping thinking/reasoning blocks."""
        if '<think>' in content and "</think>" in content:
            post_think = content.find('</think>') + len("</think>")
            content = content[post_think:].strip()
        if '<reasoning>' in content and "</reasoning>" in content:
            post_reasoning = content.find('</reasoning>') + len("</reasoning>")
            content = content[post_reasoning:].strip()
        return content

    def _adjust_payload_for_endpoint(self, payload: Dict[str, Any], model: str) -> None:
        """Adjust payload for specific API endpoints and model requirements."""
        if self.base_url == 'https://api.openai.com/v1/chat/completions':
            if 'min_p' in payload:
                del payload['min_p']
            if model == 'o3':
                del payload['max_tokens']
                payload['max_completion_tokens'] = payload.pop('max_tokens', 4000)
                payload['temperature'] = 1
            elif model in ['gpt-5-2025-08-07', 'gpt-5-mini-2025-08-07', 'gpt-5-nano-2025-08-07']:
                payload['reasoning_effort'] = "minimal"
                del payload['max_tokens']
                payload['max_completion_tokens'] = payload.pop('max_tokens', 4000)
                payload['temperature'] = 1
            elif model in ['gpt-5-chat-latest']:
                del payload['max_tokens']
                payload['max_completion_tokens'] = payload.pop('max_tokens', 4000)
                payload['temperature'] = 1

        elif self.base_url == "https://openrouter.ai/api/v1/chat/completions":
            if 'qwen3' in model.lower():
                system_msg = [{"role": "system", "content": "/no_think"}]
                payload['messages'] = system_msg + payload['messages']

            if model == 'openai/o3':
                payload["reasoning"] = {
                    "effort": "low",
                    "exclude": True
                }

    def _handle_request_error(self, e: Exception, response, attempt: int, model: str) -> None:
        """Handle request errors with appropriate logging and retry logic."""
        try:
            logging.error(response.text)
        except:
            pass
        logging.error(f"Request failed on attempt {attempt+1}/{self.max_retries} for model {model}: {e}")
        if response is not None:
            logging.error(f"Response status code: {response.status_code}")
            try:
                logging.error(f"Response body: {response.text}")
            except Exception:
                logging.error("Could not read response body.")

            if response.status_code == 429:
                logging.warning("Rate limit exceeded. Backing off...")
                delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                logging.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            elif response.status_code >= 500:
                logging.warning(f"Server error ({response.status_code}). Retrying...")
            else:
                logging.warning("API error. Retrying...")
