"""Token usage tracking for OpenAI API calls."""

import json
from dataclasses import dataclass, field
from pathlib import Path

# Pricing per 1M tokens (as of 2024)
PRICING = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

DEFAULT_BUDGET = 10.00  # $10 budget


@dataclass
class TokenUsage:
    """Track token usage for a specific model."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost(self) -> float:
        """Calculate cost for this model's usage."""
        pricing = PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class TokenTracker:
    """Track cumulative token usage and costs."""

    budget: float = DEFAULT_BUDGET
    usage_by_model: dict[str, TokenUsage] = field(default_factory=dict)
    save_path: Path | None = None

    def __post_init__(self) -> None:
        """Load existing usage if save_path exists."""
        if self.save_path and self.save_path.exists():
            self.load()

    def add_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> None:
        """Add token usage for a model.

        Args:
            model: Model name (e.g., "text-embedding-3-small").
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
        """
        if model not in self.usage_by_model:
            self.usage_by_model[model] = TokenUsage(model=model)

        self.usage_by_model[model].input_tokens += input_tokens
        self.usage_by_model[model].output_tokens += output_tokens

        if self.save_path:
            self.save()

    @property
    def total_cost(self) -> float:
        """Calculate total cost across all models."""
        return sum(u.cost for u in self.usage_by_model.values())

    @property
    def remaining_budget(self) -> float:
        """Calculate remaining budget."""
        return self.budget - self.total_cost

    @property
    def is_over_budget(self) -> bool:
        """Check if budget exceeded."""
        return self.total_cost >= self.budget

    def check_budget(self, raise_error: bool = True) -> bool:
        """Check if within budget.

        Args:
            raise_error: If True, raise exception when over budget.

        Returns:
            True if within budget.

        Raises:
            BudgetExceededError: If over budget and raise_error is True.
        """
        if self.is_over_budget:
            if raise_error:
                raise BudgetExceededError(
                    f"Budget exceeded! Spent ${self.total_cost:.4f} "
                    f"of ${self.budget:.2f} budget."
                )
            return False
        return True

    def get_summary(self) -> str:
        """Get a formatted summary of usage."""
        lines = [
            "=" * 50,
            "TOKEN USAGE SUMMARY",
            "=" * 50,
        ]

        total_input = 0
        total_output = 0

        for model, usage in sorted(self.usage_by_model.items()):
            lines.append(f"\n{model}:")
            lines.append(f"  Input tokens:  {usage.input_tokens:,}")
            lines.append(f"  Output tokens: {usage.output_tokens:,}")
            lines.append(f"  Cost:          ${usage.cost:.4f}")
            total_input += usage.input_tokens
            total_output += usage.output_tokens

        lines.append("")
        lines.append("-" * 50)
        lines.append(f"Total Input Tokens:  {total_input:,}")
        lines.append(f"Total Output Tokens: {total_output:,}")
        lines.append(f"Total Cost:          ${self.total_cost:.4f}")
        lines.append(f"Budget:              ${self.budget:.2f}")
        lines.append(f"Remaining:           ${self.remaining_budget:.4f}")

        if self.is_over_budget:
            lines.append("\n⚠️  BUDGET EXCEEDED!")
        elif self.remaining_budget < 1.0:
            lines.append(f"\n⚠️  Low budget warning: ${self.remaining_budget:.4f} left")

        lines.append("=" * 50)
        return "\n".join(lines)

    def save(self) -> None:
        """Save usage to file."""
        if not self.save_path:
            return

        data = {
            "budget": self.budget,
            "usage": {
                model: {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cost": u.cost,
                }
                for model, u in self.usage_by_model.items()
            },
            "total_cost": self.total_cost,
            "remaining_budget": self.remaining_budget,
        }

        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load usage from file."""
        if not self.save_path or not self.save_path.exists():
            return

        with open(self.save_path, "r") as f:
            data = json.load(f)

        self.budget = data.get("budget", DEFAULT_BUDGET)
        for model, usage_data in data.get("usage", {}).items():
            self.usage_by_model[model] = TokenUsage(
                model=model,
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
            )

    def reset(self) -> None:
        """Reset all usage tracking."""
        self.usage_by_model = {}
        if self.save_path:
            self.save()


class BudgetExceededError(Exception):
    """Raised when budget is exceeded."""

    pass


# Global tracker instance
_tracker: TokenTracker | None = None


def get_tracker(
    budget: float = DEFAULT_BUDGET,
    save_path: str | Path | None = "token_usage.json",
) -> TokenTracker:
    """Get or create the global token tracker.

    Args:
        budget: Maximum budget in dollars.
        save_path: Path to save usage data.

    Returns:
        TokenTracker instance.
    """
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker(
            budget=budget,
            save_path=Path(save_path) if save_path else None,
        )
    return _tracker


def track_embedding(input_tokens: int) -> None:
    """Track embedding API usage.

    Args:
        input_tokens: Number of tokens embedded.
    """
    tracker = get_tracker()
    tracker.add_usage("text-embedding-3-small", input_tokens=input_tokens)
    tracker.check_budget()


def track_chat(input_tokens: int, output_tokens: int) -> None:
    """Track chat API usage.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
    """
    tracker = get_tracker()
    tracker.add_usage(
        "gpt-4o-mini", input_tokens=input_tokens, output_tokens=output_tokens
    )
    tracker.check_budget()
