#!/usr/bin/env python3
"""Train a Deep CFR agent on Truco.

Usage:
    python scripts/train_deep_cfr.py [--config {debug,local,gcp}] [OPTIONS]

Examples:
    # Quick debug run
    python scripts/train_deep_cfr.py --config debug
    
    # Full local training
    python scripts/train_deep_cfr.py --config local --iterations 200
    
    # GCP training
    python scripts/train_deep_cfr.py --config gcp --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pyspiel

from src.config import DeepCFRConfig, get_debug_config, get_local_config, get_gcp_config
from src.trainers.deep_cfr import train_deep_cfr, DeepCFRTrainer
from src.agents.baselines import create_baseline_policies
from src.evaluation.tournament import create_evaluation_callback
from src.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep CFR agent on Truco")
    
    # Config presets
    parser.add_argument(
        "--config",
        type=str,
        choices=["debug", "local", "gcp"],
        default="local",
        help="Configuration preset",
    )
    
    # Override options
    parser.add_argument("--iterations", type=int, help="Number of training iterations")
    parser.add_argument("--traversals", type=int, help="Traversals per iteration")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output options
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/deep_cfr"),
        help="Checkpoint directory",
    )
    parser.add_argument("--eval-games", type=int, default=50, help="Games per evaluation")
    parser.add_argument("--no-eval", action="store_true", help="Disable evaluation during training")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set up logging
    logger = setup_logger(
        name="deep_cfr",
        log_file=args.checkpoint_dir / "training.log",
    )
    
    logger.info("=" * 60)
    logger.info("Deep CFR Training for Truco")
    logger.info("=" * 60)
    
    # Load config preset
    if args.config == "debug":
        config_dict = get_debug_config()
    elif args.config == "gcp":
        config_dict = get_gcp_config()
    else:
        config_dict = get_local_config()
    
    config = config_dict["deep_cfr"]
    
    # Apply overrides
    if args.iterations:
        config.num_iterations = args.iterations
    if args.traversals:
        config.num_traversals = args.traversals
    if args.lr:
        config.learning_rate = args.lr
    config.device = args.device
    config.seed = args.seed
    config.checkpoint_dir = args.checkpoint_dir
    
    # Log configuration
    logger.info(f"Configuration: {args.config}")
    logger.info(f"  Iterations: {config.num_iterations}")
    logger.info(f"  Traversals: {config.num_traversals}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Checkpoint dir: {config.checkpoint_dir}")
    
    # Create evaluation callback
    eval_callback = None
    if not args.no_eval:
        game = pyspiel.load_game("truco")
        baselines = create_baseline_policies(
            game,
            ismcts_simulations=[100],  # Quick evaluation
            seed=args.seed,
        )
        eval_callback = create_evaluation_callback(
            game, baselines, num_eval_games=args.eval_games
        )
        logger.info(f"Evaluation enabled: {args.eval_games} games vs {list(baselines.keys())}")
    
    # Train
    logger.info("Starting training...")
    trained_policy, metrics = train_deep_cfr(config, eval_callback)
    
    # Log final metrics
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Total time: {metrics['total_time']:.1f}s")
    logger.info(f"  Iterations: {metrics['iterations']}")
    if metrics.get("final_policy_loss"):
        logger.info(f"  Final policy loss: {metrics['final_policy_loss']:.6f}")
    logger.info("=" * 60)
    
    return trained_policy, metrics


if __name__ == "__main__":
    main()
