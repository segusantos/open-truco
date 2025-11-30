#!/usr/bin/env python3
"""Train MCCFR agent for Truco.

Uses OpenSpiel's fast C++ MCCFR implementation.

Example usage:
    # Quick debug run (10k iterations)
    python -m scripts.train_mccfr --config debug
    
    # Local training (1M iterations, ~1-2 minutes)
    python -m scripts.train_mccfr --config local
    
    # Full training (10M iterations)
    python -m scripts.train_mccfr --config full
"""

import argparse
import logging
from pathlib import Path

import pyspiel

from src.config import MCCFRConfig
from src.trainers.mccfr import MCCFRTrainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mccfr")


def get_config(name: str) -> MCCFRConfig:
    """Get predefined configuration by name."""
    configs = {
        "debug": MCCFRConfig(
            num_iterations=20_000,
            checkpoint_freq=5_000,
            eval_freq=5_000,
            checkpoint_dir=Path("checkpoints/mccfr_debug"),
        ),
        "local": MCCFRConfig(
            num_iterations=50_000,
            checkpoint_freq=5_000,
            eval_freq=5_000,
            checkpoint_dir=Path("checkpoints/mccfr_local"),
        ),
        "full": MCCFRConfig(
            num_iterations=10_000_000,
            checkpoint_freq=1_000_000,
            eval_freq=1_000_000,
            checkpoint_dir=Path("checkpoints/mccfr_full"),
        ),
    }
    
    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")
    
    return configs[name]


def create_eval_callback(game, num_eval_games: int = 100):
    """Create evaluation callback that plays against random."""
    import random as py_random
    import numpy as np
    
    def eval_callback(iteration, policy, metrics):
        """Evaluate policy against random opponent."""
        wins = 0
        
        for game_num in range(num_eval_games):
            state = game.new_initial_state()
            
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    actions, probs = zip(*outcomes)
                    action = py_random.choices(actions, weights=probs)[0]
                else:
                    player = state.current_player()
                    if player == 0:
                        # Our trained policy
                        action_probs = policy.action_probabilities(state)
                        actions = list(action_probs.keys())
                        probs = list(action_probs.values())
                        
                        # Sample from policy
                        action = py_random.choices(actions, weights=probs)[0]
                    else:
                        # Random opponent
                        legal = state.legal_actions()
                        action = py_random.choice(legal)
                
                state.apply_action(action)
            
            # Check if player 0 won
            returns = state.returns()
            if returns[0] > returns[1]:
                wins += 1
        
        win_rate = wins / num_eval_games
        return {"win_rate_vs_random": win_rate}
    
    return eval_callback


def main():
    parser = argparse.ArgumentParser(description="Train MCCFR agent for Truco")
    parser.add_argument(
        "--config",
        type=str,
        default="local",
        choices=["debug", "local", "full"],
        help="Configuration preset",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override number of iterations",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override checkpoint directory",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="outcome",
        choices=["outcome", "external"],
        help="MCCFR variant (outcome is faster)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation during training",
    )
    args = parser.parse_args()
    
    # Get config
    config = get_config(args.config)
    
    # Apply overrides
    if args.iterations:
        config.num_iterations = args.iterations
    if args.output_dir:
        config.checkpoint_dir = Path(args.output_dir)
    if args.solver:
        config.solver_type = args.solver
    
    # Print config
    logger.info("=" * 60)
    logger.info("MCCFR Training for Truco")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"  Solver: {config.solver_type}")
    logger.info(f"  Iterations: {config.num_iterations:,}")
    logger.info(f"  Epsilon: {config.epsilon}")
    logger.info(f"  Checkpoint dir: {config.checkpoint_dir}")
    
    # Create trainer
    game = pyspiel.load_game(config.game_name)
    
    eval_callback = None
    if not args.no_eval:
        logger.info("Evaluation: 100 games vs random every checkpoint")
        eval_callback = create_eval_callback(game, num_eval_games=100)
    
    logger.info("Starting training...")
    
    trainer = MCCFRTrainer(config, eval_callback=eval_callback)
    _, final_metrics = trainer.train()
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Total time: {final_metrics['total_time']:.1f}s")
    logger.info(f"  Speed: {final_metrics['iters_per_sec']:,.0f} iterations/sec")
    logger.info(f"  Final checkpoint: {config.checkpoint_dir}/final.pkl")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
