from __future__ import annotations

def terminal_reward(died: bool) -> float:
    # Exact same logic you currently use inline
    return -1.0 if died else 1.0


def step_reward() -> float:
    # Exact same logic you currently use for intermediate transitions
    return 0.0


def compute_reward(done: bool, died: bool) -> float:
    """
    Convenience wrapper: intermediate transitions -> 0.0,
    terminal transitions -> +/- 1.0 depending on death.
    """
    return terminal_reward(died) if done else step_reward()
