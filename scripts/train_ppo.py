#!/usr/bin/env python3
"""Train a PPO agent on Truco via self-play.

Usage:
    python scripts/train_ppo.py [--config {debug,local,gcp}] [OPTIONS]

Examples:
    # Quick debug run
    python scripts/train_ppo.py --config debug
    
    # Full local training
    python scripts/train_ppo.py --config local --timesteps 1000000
    
    # GCP training with more parallel envs
    python scripts/train_ppo.py --config gcp --num-envs 16 --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pyspiel

from src.config import PPOConfig, get_debug_config, get_local_config, get_gcp_config
from src.trainers.ppo import train_ppo, PPOTrainer
from src.agents.baselines import create_baseline_policies
from src.evaluation.tournament import create_evaluation_callback
from src.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent on Truco via self-play")
    
    # Config presets
    parser.add_argument(
        "--config",
        type=str,
        choices=["debug", "local", "gcp"],
        default="local",
        help="Configuration preset",
    )
    
    # Override options
    parser.add_argument("--timesteps", type=int, help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output options
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/ppo"),
        help="Checkpoint directory",
    )
    parser.add_argument("--eval-games", type=int, default=50, help="Games per evaluation")
    parser.add_argument("--no-eval", action="store_true", help="Disable evaluation during training")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set up logging
    logger = setup_logger(
        name="ppo",
        log_file=args.checkpoint_dir / "training.log",
    )
    
    logger.info("=" * 60)
    logger.info("PPO Self-Play Training for Truco")
    logger.info("=" * 60)
    
    # Load config preset
    if args.config == "debug":
        config_dict = get_debug_config()
    elif args.config == "gcp":
        config_dict = get_gcp_config()
    else:
        config_dict = get_local_config()
    
    config = config_dict["ppo"]
    
    # Apply overrides
    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.num_envs:
        config.num_envs = args.num_envs
    if args.lr:
        config.learning_rate = args.lr
    config.device = args.device
    config.seed = args.seed
    config.checkpoint_dir = args.checkpoint_dir
    
    # Log configuration
    logger.info(f"Configuration: {args.config}")
    logger.info(f"  Total timesteps: {config.total_timesteps}")
    logger.info(f"  Num envs: {config.num_envs}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Checkpoint dir: {config.checkpoint_dir}")
    
    # Create evaluation callback
    eval_callback = None
    if not args.no_eval:
        game = pyspiel.load_game("truco")
        baselines = create_baseline_policies(
            game,
            ismcts_simulations=[100],
            seed=args.seed,
        )
        eval_callback = create_evaluation_callback(
            game, baselines, num_eval_games=args.eval_games
        )
        logger.info(f"Evaluation enabled: {args.eval_games} games vs {list(baselines.keys())}")
    
    # Train
    logger.info("Starting training...")
    trained_policy, metrics = train_ppo(config, eval_callback)
    
    # Log final metrics
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Total time: {metrics['total_time']:.1f}s")
    logger.info(f"  Total steps: {metrics['total_steps']}")
    logger.info(f"  Updates: {metrics['updates']}")
    logger.info(f"  Final avg return: {metrics['final_avg_return']:.3f}")
    logger.info("=" * 60)
    
    return trained_policy, metrics


if __name__ == "__main__":
    main()
