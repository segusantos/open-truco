#!/usr/bin/env python3
"""Evaluate trained agents in a tournament.

Usage:
    python scripts/evaluate.py [OPTIONS]

Examples:
    # Evaluate all checkpoints in a directory
    python scripts/evaluate.py --agents checkpoints/deep_cfr/final.pt checkpoints/nfsp/final.pkl
    
    # Quick evaluation
    python scripts/evaluate.py --num-games 100
    
    # Full tournament with all baselines
    python scripts/evaluate.py --num-games 1000 --ismcts-sims 100 500 1000
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pyspiel
import torch

from src.config import EvalConfig
from src.agents.baselines import create_baseline_policies, RandomPolicy
from src.evaluation.tournament import run_tournament, head_to_head
from src.utils.logging import setup_logger
from open_spiel.python import policy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Truco agents in a tournament")
    
    # Agent checkpoints
    parser.add_argument(
        "--agents",
        type=Path,
        nargs="*",
        default=[],
        help="Paths to agent checkpoints",
    )
    
    # Baseline settings
    parser.add_argument(
        "--ismcts-sims",
        type=int,
        nargs="+",
        default=[100, 500],
        help="IS-MCTS simulation counts for baselines",
    )
    parser.add_argument("--no-baselines", action="store_true", help="Skip baseline agents")
    
    # Tournament settings
    parser.add_argument("--num-games", type=int, default=500, help="Games per matchup")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    
    return parser.parse_args()


def load_deep_cfr_policy(checkpoint_path: Path, game: pyspiel.Game):
    """Load a Deep CFR policy from checkpoint."""
    from src.trainers.deep_cfr import DeepCFRTrainer
    from src.config import DeepCFRConfig
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = DeepCFRConfig(**checkpoint.get("config", {}))
    
    trainer = DeepCFRTrainer(config)
    trainer.load_checkpoint(checkpoint_path)
    
    return trainer.get_policy()


def load_ppo_policy(checkpoint_path: Path, game: pyspiel.Game):
    """Load a PPO policy from checkpoint."""
    from src.trainers.ppo import PPOSelfPlayAgent, PPOPolicy
    from src.config import PPOConfig
    import numpy as np
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_dict = checkpoint.get("config", {})
    
    info_state_shape = game.information_state_tensor_shape()
    info_state_size = int(np.prod(info_state_shape))
    num_actions = game.num_distinct_actions()
    
    hidden_sizes = config_dict.get("hidden_layers_sizes", (256, 256))
    
    agent = PPOSelfPlayAgent(
        info_state_size=info_state_size,
        num_actions=num_actions,
        hidden_sizes=hidden_sizes,
        device="cpu",
    )
    agent.load_state_dict(checkpoint["agent_state"])
    
    return PPOPolicy(game, agent, "cpu")


def load_agent(checkpoint_path: Path, game: pyspiel.Game) -> policy.Policy:
    """Load an agent from a checkpoint file.
    
    Determines the agent type from the checkpoint structure.
    """
    path = Path(checkpoint_path)
    
    if path.suffix == ".pt":
        # Could be Deep CFR or PPO
        checkpoint = torch.load(path, map_location="cpu")
        
        if "policy_network_state" in checkpoint:
            return load_deep_cfr_policy(path, game)
        elif "agent_state" in checkpoint:
            return load_ppo_policy(path, game)
        else:
            raise ValueError(f"Unknown checkpoint format: {path}")
    
    elif path.suffix == ".pkl":
        # NFSP checkpoint
        # Note: NFSP requires JAX, so this is more complex
        raise NotImplementedError("NFSP checkpoint loading requires JAX environment")
    
    else:
        raise ValueError(f"Unknown checkpoint format: {path}")


def main():
    args = parse_args()
    
    # Set up logging
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        name="evaluate",
        log_file=args.output_dir / "evaluation.log",
    )
    
    logger.info("=" * 60)
    logger.info("Truco Agent Tournament")
    logger.info("=" * 60)
    
    # Load game
    game = pyspiel.load_game("truco")
    logger.info(f"Game: {game}")
    
    # Collect agents
    agents: Dict[str, policy.Policy] = {}
    
    # Load trained agents
    for checkpoint_path in args.agents:
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            continue
        
        name = checkpoint_path.stem
        try:
            agent_policy = load_agent(checkpoint_path, game)
            agents[name] = agent_policy
            logger.info(f"Loaded agent: {name} from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load {checkpoint_path}: {e}")
    
    # Add baseline agents
    if not args.no_baselines:
        baselines = create_baseline_policies(
            game,
            ismcts_simulations=args.ismcts_sims,
            seed=args.seed,
        )
        agents.update(baselines)
        logger.info(f"Added {len(baselines)} baseline agents")
    
    if len(agents) < 2:
        logger.error("Need at least 2 agents for tournament")
        logger.info("Using random agents as placeholders")
        agents["random_1"] = RandomPolicy(game, args.seed)
        agents["random_2"] = RandomPolicy(game, args.seed + 1)
    
    logger.info(f"Total agents: {len(agents)}")
    logger.info(f"  {list(agents.keys())}")
    logger.info(f"Games per matchup: {args.num_games}")
    
    # Run tournament
    logger.info("\nRunning tournament...")
    results = run_tournament(
        game=game,
        agents=agents,
        num_games_per_match=args.num_games,
        seed=args.seed,
        verbose=True,
    )
    
    # Save results
    import json
    
    results_dict = {
        "rankings": results.compute_rankings(),
        "elo_ratings": {
            name: {"rating": r.rating, "record": r.record}
            for name, r in results.elo_ratings.items()
        },
        "matches": {
            f"{a}_vs_{b}": stats.to_dict()
            for (a, b), stats in results.match_stats.items()
        },
    }
    
    results_file = args.output_dir / "tournament_results.json"
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    logger.info("\n" + results.summary())
    
    return results


if __name__ == "__main__":
    main()
