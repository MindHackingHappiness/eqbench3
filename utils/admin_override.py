"""Admin override system for EQBench3 expensive operations."""
import os
import logging
import getpass
from typing import Optional

class AdminOverrideManager:
    """Manages admin overrides for expensive EQBench3 operations."""

    def __init__(self):
        self.override_key = os.getenv("EQBENCH_ADMIN_OVERRIDE", "0")
        self.max_tokens_per_batch = int(os.getenv("EQBENCH_MAX_TOKENS_PER_BATCH", "30000"))
        self.allow_expensive_models = os.getenv("EQBENCH_ALLOW_EXPENSIVE_MODELS", "0").lower() == "1"
        self.require_confirmation = not self.allow_expensive_models

    def check_override_for_large_payload(self, model: str, estimated_tokens: int, operation: str = "evaluation") -> bool:
        """
        Check if admin override is required for large payloads.
        Returns True if operation is allowed, False if rejected.
        """
        # Automatic approval for cheap models
        if "1.5-flash" in model:
            return True

        # Automatic rejection for very expensive combinations
        if estimated_tokens > 75000 and ("2.5" in model or "2.0" in model):
            logging.warning(f"REJECTED: {operation} with {model} would cost ~${estimated_tokens/1000000*15:.2f}")
            return False

        # Require override for expensive combinations
        if estimated_tokens > self.max_tokens_per_batch or ("2.5" in model and estimated_tokens > 20000):
            return self._get_admin_approval(model, estimated_tokens, operation)

        return True

    def _get_admin_approval(self, model: str, estimated_tokens: int, operation: str) -> bool:
        """Get admin approval for expensive operation."""
        if self.override_key == "1":
            logging.info(f"ADMIN OVERRIDE: Auto-approved {operation} with {model} ({estimated_tokens} tokens)")
            return True

        try:
            estimated_cost = (estimated_tokens / 1000000) * 30  # Rough estimate for expensive models

            print(f"\n{'='*60}")
            print("⚠️  EXPENSIVE OPERATION REQUIRES APPROVAL")
            print(f"{'='*60}")
            print(f"Operation: {operation}")
            print(f"Model: {model}")
            print(f"Estimated tokens: {estimated_tokens:,}")
            print(f"Estimated cost: ${estimated_cost:.2f}")
            print(f"{'='*60}")

            response = input("Allow this operation? (yes/no): ").strip().lower()

            if response in ["yes", "y"]:
                logging.info(f"ADMIN APPROVED: {operation} with {model} ({estimated_tokens} tokens)")
                return True
            else:
                logging.info(f"ADMIN REJECTED: {operation} with {model} ({estimated_tokens} tokens)")
                return False

        except KeyboardInterrupt:
            logging.info("Operation cancelled by user")
            return False
        except Exception as e:
            logging.error(f"Admin approval failed: {e}")
            return False

    def log_operation(self, model: str, estimated_tokens: int, operation: str, approved: bool):
        """Log all override decisions for audit trail."""
        status = "APPROVED" if approved else "REJECTED"
        logging.info(f"EQBENCH_ADMIN_CHECK: {status} | {operation} | {model} | {estimated_tokens} tokens")

class BatchManager:
    """Manages batch sizes and payload limits."""

    def __init__(self):
        self.max_batch_size = int(os.getenv("EQBENCH_MAX_BATCH_SIZE", "30"))
        self.max_tokens_total = int(os.getenv("EQBENCH_MAX_TOTAL_TOKENS", "100000"))

    def enforce_batch_limits(self, batch_size: int, model: str) -> int:
        """Enforce batch size limits based on model."""
        if "2.5" in model and batch_size > 10:
            logging.warning(f"Reducing batch size from {batch_size} to 10 for expensive model {model}")
            return 10
        elif batch_size > self.max_batch_size:
            logging.warning(f"Reducing batch size from {batch_size} to {self.max_batch_size}")
            return self.max_batch_size
        return batch_size
