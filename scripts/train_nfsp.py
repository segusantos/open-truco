#!/usr/bin/env python3
"""Train an NFSP agent on Truco.

Usage:
    python scripts/train_nfsp.py [--config {debug,local,gcp}] [OPTIONS]

Examples:
    # Quick debug run
    python scripts/train_nfsp.py --config debug
    
    # Full local training
    python scripts/train_nfsp.py --config local --episodes 1000000
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pyspiel

from src.config import NFSPConfig, get_debug_config, get_local_config, get_gcp_config
from src.trainers.nfsp import train_nfsp, NFSPTrainer
from src.agents.baselines import create_baseline_policies
from src.evaluation.tournament import create_evaluation_callback
from src.utils.logging import setup_logger


if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
logger = logging.getLogger("nfsp")


def parse_args():
    parser = argparse.ArgumentParser(description="Train NFSP agent on Truco")
    
    # Config presets
    parser.add_argument(
        "--config",
        type=str,
        choices=["debug", "local", "gcp"],
        default="local",
        help="Configuration preset",
    )
    
    # Override options
    parser.add_argument("--episodes", type=int, help="Number of training episodes")
    parser.add_argument("--hidden-size", type=int, help="Hidden layer size")
    parser.add_argument("--anticipatory", type=float, help="Anticipatory parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output options
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/nfsp"),
        help="Checkpoint directory",
    )
    parser.add_argument("--eval-games", type=int, default=50, help="Games per evaluation")
    parser.add_argument("--no-eval", action="store_true", help="Disable evaluation during training")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set up logging
    logger = setup_logger(
        name="nfsp",
        log_file=args.checkpoint_dir / "training.log",
    )
    
    logger.info("=" * 60)
    logger.info("NFSP Training for Truco")
    logger.info("=" * 60)
    
    # Load config preset
    if args.config == "debug":
        config_dict = get_debug_config()
    elif args.config == "gcp":
        config_dict = get_gcp_config()
    else:
        config_dict = get_local_config()
    
    config = config_dict["nfsp"]
    
    # Apply overrides
    if args.episodes:
        config.num_train_episodes = args.episodes
    if args.hidden_size:
        config.hidden_layers_sizes = (args.hidden_size, args.hidden_size)
    if args.anticipatory:
        config.anticipatory_param = args.anticipatory
    config.seed = args.seed
    config.checkpoint_dir = args.checkpoint_dir
    
    # Log configuration
    logger.info(f"Configuration: {args.config}")
    logger.info(f"  Episodes: {config.num_train_episodes}")
    logger.info(f"  Hidden layers: {config.hidden_layers_sizes}")
    logger.info(f"  Anticipatory param: {config.anticipatory_param}")
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
    trained_policy, metrics = train_nfsp(config, eval_callback)
    
    # Log final metrics
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Total time: {metrics['total_time']:.1f}s")
    logger.info(f"  Episodes: {metrics['episodes']}")
    logger.info("=" * 60)
    
    return trained_policy, metrics


if __name__ == "__main__":
    main()
