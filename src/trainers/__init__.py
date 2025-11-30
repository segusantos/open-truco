# Training modules for Truco agents
# Use lazy imports to avoid requiring all dependencies when only using one trainer

__all__ = ["train_deep_cfr", "train_nfsp", "train_ppo"]


def __getattr__(name):
    """Lazy import trainers to avoid dependency issues."""
    if name == "train_deep_cfr":
        from src.trainers.deep_cfr import train_deep_cfr
        return train_deep_cfr
    elif name == "train_nfsp":
        from src.trainers.nfsp import train_nfsp
        return train_nfsp
    elif name == "train_ppo":
        from src.trainers.ppo import train_ppo
        return train_ppo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
