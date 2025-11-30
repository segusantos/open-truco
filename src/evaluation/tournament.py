"""Tournament and head-to-head evaluation for Truco agents.

Implements match simulation, round-robin tournaments, and
agent comparison utilities.
"""

import time
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import logging

import numpy as np
import pyspiel

from open_spiel.python import policy

from src.evaluation.metrics import (
    MatchStatistics,
    TournamentResults,
    compute_elo,
    EloRating,
)
from src.config import EvalConfig


logger = logging.getLogger(__name__)


def play_game(
    game: pyspiel.Game,
    policies: List[policy.Policy],
    seed: Optional[int] = None,
) -> Tuple[List[float], int]:
    """Play a single game between two policies.
    
    Args:
        game: The game to play.
        policies: List of policies, one per player.
        seed: Optional random seed.
        
    Returns:
        Tuple of (returns, num_actions).
    """
    if seed is not None:
        np.random.seed(seed)
    
    state = game.new_initial_state()
    num_actions = 0
    
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np.random.choice(actions, p=probs)
            state.apply_action(action)
        else:
            current_player = state.current_player()
            action_probs = policies[current_player].action_probabilities(state)
            
            # Sample action from policy
            actions = list(action_probs.keys())
            probs = [action_probs[a] for a in actions]
            
            # Normalize probabilities
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                probs = [1.0 / len(actions)] * len(actions)
            
            action = np.random.choice(actions, p=probs)
            state.apply_action(action)
            num_actions += 1
    
    return state.returns(), num_actions


def head_to_head(
    game: pyspiel.Game,
    policy_a: policy.Policy,
    policy_b: policy.Policy,
    num_games: int,
    name_a: str = "Agent A",
    name_b: str = "Agent B",
    seed: int = 42,
    alternate_start: bool = True,
    verbose: bool = False,
) -> MatchStatistics:
    """Run head-to-head evaluation between two policies.
    
    Args:
        game: The game to play.
        policy_a: First policy.
        policy_b: Second policy.
        num_games: Number of games to play.
        name_a: Name for first agent.
        name_b: Name for second agent.
        seed: Random seed.
        alternate_start: Whether to alternate starting player.
        verbose: Whether to log progress.
        
    Returns:
        MatchStatistics with results.
    """
    np.random.seed(seed)
    
    wins_a = 0
    wins_b = 0
    draws = 0
    total_points_a = 0.0
    total_points_b = 0.0
    total_actions = 0
    
    for i in range(num_games):
        # Alternate starting positions
        if alternate_start and i % 2 == 1:
            policies = [policy_b, policy_a]
            player_a_idx = 1
        else:
            policies = [policy_a, policy_b]
            player_a_idx = 0
        
        returns, num_actions = play_game(game, policies, seed=seed + i)
        total_actions += num_actions
        
        return_a = returns[player_a_idx]
        return_b = returns[1 - player_a_idx]
        
        total_points_a += return_a
        total_points_b += return_b
        
        if return_a > return_b:
            wins_a += 1
        elif return_b > return_a:
            wins_b += 1
        else:
            draws += 1
        
        if verbose and (i + 1) % max(1, num_games // 10) == 0:
            logger.info(f"  Game {i + 1}/{num_games}: {wins_a}-{wins_b}-{draws}")
    
    avg_game_length = total_actions / num_games if num_games > 0 else 0.0
    
    return MatchStatistics(
        player_a=name_a,
        player_b=name_b,
        num_games=num_games,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
        total_points_a=total_points_a,
        total_points_b=total_points_b,
        avg_game_length=avg_game_length,
    )


def run_tournament(
    game: pyspiel.Game,
    agents: Dict[str, policy.Policy],
    num_games_per_match: int = 100,
    seed: int = 42,
    k_factor: float = 32.0,
    verbose: bool = True,
) -> TournamentResults:
    """Run a round-robin tournament between all agents.
    
    Args:
        game: The game to play.
        agents: Dictionary mapping agent names to policies.
        num_games_per_match: Number of games per matchup.
        seed: Random seed.
        k_factor: K-factor for Elo updates.
        verbose: Whether to log progress.
        
    Returns:
        TournamentResults with all match statistics and Elo ratings.
    """
    agent_names = list(agents.keys())
    num_agents = len(agent_names)
    
    if verbose:
        logger.info(f"Running tournament with {num_agents} agents")
        logger.info(f"  Agents: {agent_names}")
        logger.info(f"  Games per match: {num_games_per_match}")
    
    results = TournamentResults(agents=agent_names)
    match_results: List[Tuple[str, str, int]] = []
    
    start_time = time.time()
    
    # Round-robin: each pair plays
    for i, name_a in enumerate(agent_names):
        for j, name_b in enumerate(agent_names[i + 1:], i + 1):
            if verbose:
                logger.info(f"Match: {name_a} vs {name_b}")
            
            stats = head_to_head(
                game=game,
                policy_a=agents[name_a],
                policy_b=agents[name_b],
                num_games=num_games_per_match,
                name_a=name_a,
                name_b=name_b,
                seed=seed + i * 1000 + j,
                verbose=verbose,
            )
            
            results.add_match(stats)
            
            # Record individual game results for Elo
            for _ in range(stats.wins_a):
                match_results.append((name_a, name_b, 1))
            for _ in range(stats.wins_b):
                match_results.append((name_a, name_b, -1))
            for _ in range(stats.draws):
                match_results.append((name_a, name_b, 0))
            
            if verbose:
                logger.info(f"  Result: {stats}")
    
    # Compute Elo ratings
    results.elo_ratings = compute_elo(match_results, k_factor=k_factor)
    
    elapsed = time.time() - start_time
    
    if verbose:
        logger.info(f"Tournament completed in {elapsed:.1f}s")
        logger.info(results.summary())
    
    return results


def evaluate_against_baselines(
    game: pyspiel.Game,
    trained_policy: policy.Policy,
    policy_name: str,
    baseline_policies: Dict[str, policy.Policy],
    num_games: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, MatchStatistics]:
    """Evaluate a trained policy against multiple baselines.
    
    Args:
        game: The game to play.
        trained_policy: The policy to evaluate.
        policy_name: Name for the trained policy.
        baseline_policies: Dictionary of baseline policies.
        num_games: Number of games per baseline.
        seed: Random seed.
        verbose: Whether to log progress.
        
    Returns:
        Dictionary mapping baseline names to MatchStatistics.
    """
    results = {}
    
    for baseline_name, baseline_policy in baseline_policies.items():
        if verbose:
            logger.info(f"Evaluating {policy_name} vs {baseline_name}")
        
        stats = head_to_head(
            game=game,
            policy_a=trained_policy,
            policy_b=baseline_policy,
            num_games=num_games,
            name_a=policy_name,
            name_b=baseline_name,
            seed=seed,
            verbose=verbose,
        )
        
        results[baseline_name] = stats
        
        if verbose:
            logger.info(f"  {stats}")
    
    return results


def create_evaluation_callback(
    game: pyspiel.Game,
    baseline_policies: Dict[str, policy.Policy],
    num_eval_games: int = 50,
) -> Callable:
    """Create an evaluation callback for use during training.
    
    Args:
        game: The game to play.
        baseline_policies: Baseline policies to evaluate against.
        num_eval_games: Number of games per evaluation.
        
    Returns:
        Callback function (iteration, policy, metrics) -> eval_metrics.
    """
    def eval_callback(
        iteration: int,
        trained_policy: policy.Policy,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluation callback."""
        eval_metrics = {}
        
        for baseline_name, baseline_policy in baseline_policies.items():
            stats = head_to_head(
                game=game,
                policy_a=trained_policy,
                policy_b=baseline_policy,
                num_games=num_eval_games,
                name_a="trained",
                name_b=baseline_name,
                seed=iteration,
                verbose=False,
            )
            
            eval_metrics[f"win_rate_vs_{baseline_name}"] = stats.win_rate_a
            eval_metrics[f"games_vs_{baseline_name}"] = num_eval_games
        
        return eval_metrics
    
    return eval_callback
