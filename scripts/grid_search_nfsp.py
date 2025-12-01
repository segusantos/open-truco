#!/usr/bin/env python3
"""Grid search for NFSP hyperparameters on Truco.

Usage:
    python scripts/grid_search_nfsp.py [--config {debug,local,gcp}] [OPTIONS]

Examples:
    # Quick debug grid search
    python scripts/grid_search_nfsp.py --config debug

    # Full local grid search
    python scripts/grid_search_nfsp.py --config local
"""

import argparse
import logging
import sys
from itertools import product
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pyspiel

from src.config import NFSPConfig, get_debug_config, get_local_config, get_gcp_config
from src.trainers.nfsp import train_nfsp
from src.utils.logging import setup_logger
from src.evaluation.tournament import create_evaluation_callback
from src.agents.baselines import create_baseline_policies

# Set up logging
logger = setup_logger("grid_search_nfsp")

def parse_args():
    parser = argparse.ArgumentParser(description="Grid search NFSP hyperparameters on Truco")

    # Config presets
    parser.add_argument(
        "--config",
        type=str,
        choices=["debug", "local", "gcp"],
        default="local",
        help="Configuration preset",
    )

    # Output options
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/grid_search"),
        help="Directory to save grid search results",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Set up logging
    log_file = args.results_dir / "grid_search.log"
    logger = setup_logger("grid_search_nfsp")

    # Add file handler to logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Ensure all relevant loggers propagate to the root logger
    logging.getLogger("src.trainers.nfsp").addHandler(file_handler)
    logging.getLogger("src.trainers.nfsp").setLevel(logging.INFO)

    logger.info("=" * 60)
    logger.info("Grid Search for NFSP Hyperparameters")
    logger.info("=" * 60)

    # Load config preset
    if args.config == "debug":
        config_dict = get_debug_config()
    elif args.config == "gcp":
        config_dict = get_gcp_config()
    else:
        config_dict = get_local_config()

    base_config = config_dict["nfsp"]

    # Define hyperparameter grid
    hidden_layer_sizes_grid = [(16, 16), (32, 32), (64, 64)]  # Example sizes
    anticipatory_param_grid = [0.1, 0.5, 0.9]  # Example values

    # Perform grid search
    best_metrics = None
    best_params = None

    for hidden_layers, anticipatory in product(
        hidden_layer_sizes_grid, anticipatory_param_grid
    ):
        # Update config
        # Create a copy of the base config and update only the necessary fields
        config = NFSPConfig(
            **{k: v for k, v in base_config.__dict__.items() if k not in ["hidden_layers_sizes", "anticipatory_param"]},
            hidden_layers_sizes=hidden_layers,
            anticipatory_param=anticipatory,
        )

        logger.info(f"Testing configuration: hidden_layers={hidden_layers}, anticipatory={anticipatory}")

        # Create baseline policies
        baseline_policies = create_baseline_policies(
            game=pyspiel.load_game("truco"),
            ismcts_simulations=[100],
            seed=42,
        )

        # Create evaluation callback
        eval_callback = create_evaluation_callback(
            game=pyspiel.load_game("truco"),
            baseline_policies=baseline_policies,
            num_eval_games=50,
        )

        # Train model
        trained_policy, metrics = train_nfsp(
            config=config,
            eval_callback=eval_callback,
        )

        # Log evaluation metrics
        if metrics:
            logger.info(f"Evaluation metrics for config {hidden_layers}, {anticipatory}:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")

        # Save results
        result_path = args.results_dir / f"results_{hidden_layers}_{anticipatory}.txt"
        with open(result_path, "w") as f:
            f.write(str(metrics))

        # Track best metrics
        if best_metrics is None or metrics["total_time"] < best_metrics["total_time"]:
            best_metrics = metrics
            best_params = (hidden_layers, anticipatory)

    logger.info("=" * 60)
    logger.info("Grid Search Complete!")
    logger.info(f"Best configuration: hidden_layers={best_params[0]}, anticipatory={best_params[1]}")
    logger.info(f"Best metrics: {best_metrics}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()