"""Configuration dataclasses for Truco RL training experiments.

Defines hyperparameters for MCCFR, NFSP, PPO, and evaluation.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from pathlib import Path


@dataclass
class MCCFRConfig:
    """Configuration for MCCFR (Monte Carlo CFR) training.
    
    Uses OpenSpiel's fast C++ implementation. Much faster than Deep CFR
    for large games like Truco.
    """
    # Game settings
    game_name: str = "truco"
    
    # Solver type: "outcome" (fastest) or "external"
    solver_type: str = "outcome"
    
    # Training iterations (can do millions quickly)
    num_iterations: int = 1_000_000
    
    # Exploration parameter for outcome sampling
    epsilon: float = 0.6
    
    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/mccfr"))
    checkpoint_freq: int = 100_000
    
    # Evaluation
    eval_freq: int = 100_000
    
    # Random seed
    seed: int = 42


@dataclass
class DeepCFRConfig:
    """Configuration for Deep CFR training.
    
    NOTE: Deep CFR is very slow for Truco due to the large game tree.
    Consider using MCCFRConfig instead which uses fast C++ solvers.
    """
    # Game settings
    game_name: str = "truco"
    
    # Training iterations
    num_iterations: int = 100
    num_traversals: int = 100
    
    # Network architecture
    policy_network_layers: Tuple[int, ...] = (256, 256)
    advantage_network_layers: Tuple[int, ...] = (256, 256)
    
    # Optimization
    learning_rate: float = 1e-3
    batch_size_advantage: Optional[int] = 2048
    batch_size_strategy: Optional[int] = 2048
    
    # Memory
    memory_capacity: int = int(1e7)
    
    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/deep_cfr"))
    checkpoint_freq: int = 10  # Save every N iterations
    
    # Logging
    eval_freq: int = 10  # Evaluate every N iterations
    log_freq: int = 1    # Log every N iterations
    
    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    
    # Random seed
    seed: int = 42


@dataclass
class NFSPConfig:
    """Configuration for Neural Fictitious Self-Play training.
    
    NFSP combines reinforcement learning (best response) with
    supervised learning (average policy) to converge to Nash equilibrium.
    """
    # Game settings
    game_name: str = "truco"
    
    # Training episodes
    num_train_episodes: int = int(3e6)
    eval_every: int = 10000
    
    # Network architecture
    hidden_layers_sizes: Tuple[int, ...] = (256, 256)
    
    # Buffer sizes
    replay_buffer_capacity: int = int(2e5)
    reservoir_buffer_capacity: int = int(2e6)
    
    # NFSP-specific
    anticipatory_param: float = 0.1  # Probability of using RL policy
    
    # Epsilon schedule for exploration
    epsilon_start: float = 0.06
    epsilon_end: float = 0.001
    epsilon_decay_duration: Optional[int] = None  # Defaults to num_train_episodes
    
    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/nfsp"))
    checkpoint_freq: int = 50000
    
    # Logging
    log_freq: int = 1000
    
    # Random seed
    seed: int = 42


@dataclass
class PPOConfig:
    """Configuration for PPO self-play training.
    
    PPO is used as a model-free baseline. Both players share the same
    policy network (self-play) to learn a strategy.
    """
    # Game settings
    game_name: str = "truco"
    
    # Training
    total_timesteps: int = int(1e6)
    num_envs: int = 8  # Parallel environments
    num_steps: int = 128  # Steps per rollout
    
    # Network architecture
    hidden_layers_sizes: Tuple[int, ...] = (256, 256)
    
    # PPO hyperparameters
    learning_rate: float = 2.5e-4
    gamma: float = 1.0  # No discounting for episodic games
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training schedule
    num_minibatches: int = 4
    update_epochs: int = 4
    
    # Normalization
    norm_adv: bool = True
    clip_vloss: bool = True
    
    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/ppo"))
    checkpoint_freq: int = 10000
    
    # Logging
    log_freq: int = 1000
    eval_freq: int = 10000
    
    # Device
    device: str = "cuda"
    
    # Random seed
    seed: int = 42


@dataclass
class EvalConfig:
    """Configuration for agent evaluation."""
    # Game settings
    game_name: str = "truco"
    
    # Tournament settings
    num_matches: int = 1000  # Games per matchup
    
    # IS-MCTS baseline settings
    ismcts_simulations: List[int] = field(default_factory=lambda: [100, 500, 1000])
    ismcts_uct_c: float = 2.0
    
    # Elo settings
    initial_elo: float = 1500.0
    k_factor: float = 32.0
    
    # Output
    results_dir: Path = field(default_factory=lambda: Path("results"))
    
    # Random seed
    seed: int = 42


# Preset configurations for different compute budgets

def get_debug_config() -> dict:
    """Quick debug configuration for testing."""
    return {
        "deep_cfr": DeepCFRConfig(
            num_iterations=5,
            num_traversals=10,
            policy_network_layers=(32, 32),
            advantage_network_layers=(32, 32),
            memory_capacity=int(1e4),
            eval_freq=1,
        ),
        "nfsp": NFSPConfig(
            num_train_episodes=1000,
            eval_every=100,
            hidden_layers_sizes=(32, 32),
            replay_buffer_capacity=int(1e3),
            reservoir_buffer_capacity=int(1e4),
        ),
        "ppo": PPOConfig(
            total_timesteps=10000,
            num_envs=2,
            hidden_layers_sizes=(32, 32),
            eval_freq=1000,
        ),
        "eval": EvalConfig(
            num_matches=100,
            ismcts_simulations=[10, 50],
        ),
    }


def get_local_config() -> dict:
    """Configuration for local GPU training (GTX-level)."""
    return {
        "deep_cfr": DeepCFRConfig(
            num_iterations=100,
            num_traversals=100,
            policy_network_layers=(128, 128),
            advantage_network_layers=(128, 128),
        ),
        "nfsp": NFSPConfig(
            num_train_episodes=int(1e6),
            hidden_layers_sizes=(128, 128),
        ),
        "ppo": PPOConfig(
            total_timesteps=int(5e5),
            hidden_layers_sizes=(128, 128),
        ),
        "eval": EvalConfig(
            num_matches=500,
        ),
    }


def get_gcp_config() -> dict:
    """Configuration for GCP cloud training (full scale)."""
    return {
        "deep_cfr": DeepCFRConfig(
            num_iterations=500,
            num_traversals=200,
            policy_network_layers=(256, 256),
            advantage_network_layers=(256, 256),
            memory_capacity=int(2e7),
        ),
        "nfsp": NFSPConfig(
            num_train_episodes=int(5e6),
            hidden_layers_sizes=(256, 256),
            replay_buffer_capacity=int(5e5),
            reservoir_buffer_capacity=int(5e6),
        ),
        "ppo": PPOConfig(
            total_timesteps=int(2e6),
            num_envs=16,
            hidden_layers_sizes=(256, 256),
        ),
        "eval": EvalConfig(
            num_matches=2000,
            ismcts_simulations=[100, 500, 1000, 2000],
        ),
    }
