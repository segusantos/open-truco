"""Baseline agents for Truco evaluation.

Provides random agent, IS-MCTS agent, and policy wrappers
for use as evaluation baselines.
"""

from typing import Dict, Optional, List
import logging

import numpy as np
import pyspiel

from open_spiel.python import policy
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.ismcts import ISMCTSBot


logger = logging.getLogger(__name__)


class RandomPolicy(policy.Policy):
    """Uniform random policy over legal actions."""
    
    def __init__(self, game: pyspiel.Game, seed: int = 42):
        """Initialize random policy.
        
        Args:
            game: The game to play.
            seed: Random seed.
        """
        super().__init__(game, list(range(game.num_players())))
        self._game = game
        self._rng = np.random.RandomState(seed)
    
    def action_probabilities(self, state, player_id=None):
        """Returns uniform distribution over legal actions."""
        legal_actions = state.legal_actions(state.current_player())
        num_actions = len(legal_actions)
        
        if num_actions == 0:
            return {}
        
        prob = 1.0 / num_actions
        return {action: prob for action in legal_actions}


class ISMCTSPolicy(policy.Policy):
    """Policy wrapper around IS-MCTS bot.
    
    Information Set Monte Carlo Tree Search is particularly well-suited
    for imperfect information games like Truco, as it samples from
    consistent belief states.
    """
    
    def __init__(
        self,
        game: pyspiel.Game,
        num_simulations: int = 1000,
        uct_c: float = 2.0,
        max_world_samples: int = 100,
        seed: int = 42,
    ):
        """Initialize IS-MCTS policy.
        
        Args:
            game: The game to play.
            num_simulations: Number of MCTS simulations per move.
            uct_c: UCT exploration constant.
            max_world_samples: Maximum number of world samples per node.
            seed: Random seed.
        """
        super().__init__(game, list(range(game.num_players())))
        self._game = game
        self._num_simulations = num_simulations
        self._uct_c = uct_c
        self._max_world_samples = max_world_samples
        self._seed = seed
        
        # Create bots for each player
        self._bots = [
            self._create_bot(player_id) 
            for player_id in range(game.num_players())
        ]
    
    def _create_bot(self, player_id: int) -> ISMCTSBot:
        """Create an IS-MCTS bot for a player."""
        evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=np.random.RandomState(self._seed))
        
        return ISMCTSBot(
            game=self._game,
            uct_c=self._uct_c,
            max_simulations=self._num_simulations,
            evaluator=evaluator,
            random_state=np.random.RandomState(self._seed + player_id),
        )
    
    def action_probabilities(self, state, player_id=None):
        """Returns action probabilities from IS-MCTS.
        
        Note: IS-MCTS returns a single action recommendation, not a
        distribution. We return a deterministic distribution over
        the recommended action.
        """
        current_player = state.current_player()
        legal_actions = state.legal_actions(current_player)
        
        if len(legal_actions) == 1:
            return {legal_actions[0]: 1.0}
        
        # Get IS-MCTS recommendation
        bot = self._bots[current_player]
        
        # We need to create a fresh state for the bot
        # since IS-MCTS may modify internal state
        action = bot.step(state)
        
        # Return deterministic distribution
        probs = {a: 0.0 for a in legal_actions}
        probs[action] = 1.0
        
        return probs
    
    def step(self, state) -> int:
        """Return the recommended action directly."""
        current_player = state.current_player()
        return self._bots[current_player].step(state)


class PolicyAgent:
    """Agent wrapper around a policy for step-based interfaces."""
    
    def __init__(
        self,
        policy: policy.Policy,
        player_id: int,
        seed: int = 42,
    ):
        """Initialize policy agent.
        
        Args:
            policy: The policy to wrap.
            player_id: Player ID for this agent.
            seed: Random seed.
        """
        self._policy = policy
        self._player_id = player_id
        self._rng = np.random.RandomState(seed)
    
    def step(self, state) -> int:
        """Sample an action from the policy."""
        action_probs = self._policy.action_probabilities(state, self._player_id)
        actions = list(action_probs.keys())
        probs = [action_probs[a] for a in actions]
        
        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / len(actions)] * len(actions)
        
        return self._rng.choice(actions, p=probs)


def create_random_agent(game: pyspiel.Game, seed: int = 42) -> RandomPolicy:
    """Create a random baseline agent.
    
    Args:
        game: The game to play.
        seed: Random seed.
        
    Returns:
        Random policy.
    """
    return RandomPolicy(game, seed)


def create_ismcts_agent(
    game: pyspiel.Game,
    num_simulations: int = 1000,
    uct_c: float = 2.0,
    seed: int = 42,
) -> ISMCTSPolicy:
    """Create an IS-MCTS baseline agent.
    
    Args:
        game: The game to play.
        num_simulations: Number of simulations per move.
        uct_c: UCT exploration constant.
        seed: Random seed.
        
    Returns:
        IS-MCTS policy.
    """
    return ISMCTSPolicy(
        game=game,
        num_simulations=num_simulations,
        uct_c=uct_c,
        seed=seed,
    )


def create_baseline_policies(
    game: pyspiel.Game,
    ismcts_simulations: List[int] = [100, 500, 1000],
    seed: int = 42,
) -> Dict[str, policy.Policy]:
    """Create a suite of baseline policies for evaluation.
    
    Args:
        game: The game to play.
        ismcts_simulations: List of simulation counts for IS-MCTS variants.
        seed: Random seed.
        
    Returns:
        Dictionary mapping baseline names to policies.
    """
    baselines = {
        "random": create_random_agent(game, seed),
    }
    
    for num_sims in ismcts_simulations:
        name = f"ismcts_{num_sims}"
        baselines[name] = create_ismcts_agent(
            game=game,
            num_simulations=num_sims,
            seed=seed,
        )
    
    logger.info(f"Created {len(baselines)} baseline policies: {list(baselines.keys())}")
    
    return baselines


# Convenience aliases
RandomAgent = RandomPolicy
ISMCTSAgent = ISMCTSPolicy
