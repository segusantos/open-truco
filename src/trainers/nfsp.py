"""NFSP (Neural Fictitious Self-Play) trainer for Truco.

Wraps OpenSpiel's NFSP implementation with training loop,
checkpointing, evaluation, and logging for Truco experiments.
"""

import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
import pickle

import numpy as np
import pyspiel

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.jax import nfsp

from src.config import NFSPConfig
from src.utils.logging import setup_logger


# Set up logging
logger = setup_logger(__name__)


class NFSPPolicies(policy.Policy):
    """Joint policy for NFSP agents to be used for evaluation."""
    
    def __init__(self, env, nfsp_agents, mode):
        """Initialize NFSP policies wrapper.
        
        Args:
            env: OpenSpiel RL environment.
            nfsp_agents: List of NFSP agents (one per player).
            mode: nfsp.MODE.average_policy or nfsp.MODE.best_response
        """
        game = env.game
        player_ids = list(range(game.num_players()))
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_agents
        self._mode = mode
        self._obs = {
            "info_state": [None] * game.num_players(),
            "legal_actions": [None] * game.num_players(),
        }
    
    def action_probabilities(self, state, player_id=None):
        """Returns action probabilities for the given state."""
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        
        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = state.information_state_tensor(cur_player)
        self._obs["legal_actions"][cur_player] = legal_actions
        
        info_state = rl_environment.TimeStep(
            observations=self._obs,
            rewards=None,
            discounts=None,
            step_type=None,
        )
        
        with self._policies[cur_player].temp_mode_as(self._mode):
            step_output = self._policies[cur_player].step(info_state, is_evaluation=True)
            probs = step_output.probs
        
        return {action: probs[action] for action in legal_actions}


class NFSPTrainer:
    """Trainer wrapper for NFSP on Truco."""
    
    def __init__(
        self,
        config: NFSPConfig,
        eval_callback: Optional[Callable] = None,
    ):
        """Initialize NFSP trainer.
        
        Args:
            config: Training configuration.
            eval_callback: Optional callback for evaluation.
        """
        self.config = config
        self.eval_callback = eval_callback
        
        # Set seed
        np.random.seed(config.seed)
        
        # Create environment
        self.env = rl_environment.Environment(config.game_name)
        self.game = self.env.game
        
        logger.info(f"Loaded game: {config.game_name}")
        logger.info(f"  Num players: {self.game.num_players()}")
        logger.info(f"  Num actions: {self.env.action_spec()['num_actions']}")
        
        # Get state/action sizes
        self.info_state_size = self.env.observation_spec()["info_state"][0]
        self.num_actions = self.env.action_spec()["num_actions"]
        
        # Epsilon decay duration
        epsilon_decay = config.epsilon_decay_duration or config.num_train_episodes
        
        # Initialize NFSP agents
        kwargs = {
            "replay_buffer_capacity": config.replay_buffer_capacity,
            "epsilon_decay_duration": epsilon_decay,
            "epsilon_start": config.epsilon_start,
            "epsilon_end": config.epsilon_end,
        }
        
        self.agents = [
            nfsp.NFSP(
                player_id=idx,
                state_representation_size=self.info_state_size,
                num_actions=self.num_actions,
                hidden_layers_sizes=list(config.hidden_layers_sizes),
                reservoir_buffer_capacity=config.reservoir_buffer_capacity,
                anticipatory_param=config.anticipatory_param,
                **kwargs,
            )
            for idx in range(self.game.num_players())
        ]
        
        # Training state
        self.episode = 0
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Tuple[policy.Policy, Dict[str, Any]]:
        """Run full training loop.
        
        Returns:
            Trained average policy and final metrics.
        """
        logger.info(f"Starting NFSP training for {self.config.num_train_episodes} episodes")
        start_time = time.time()
        
        losses_history = []
        
        for episode in range(1, self.config.num_train_episodes + 1):
            self.episode = episode
            
            # Run one episode
            time_step = self.env.reset()
            
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = self.agents[player_id].step(time_step)
                time_step = self.env.step([agent_output.action])
            
            # Episode is over - step all agents with final state
            for agent in self.agents:
                agent.step(time_step)
            
            # Log metrics
            if episode % self.config.log_freq == 0:
                losses = [agent.loss for agent in self.agents]
                losses_history.append(losses)
                
                logger.info(
                    f"Episode {episode}/{self.config.num_train_episodes} | "
                    f"Losses: {losses}"
                )
            
            # Evaluate
            if episode % self.config.eval_every == 0:
                logger.info(f"Evaluating at episode {episode}")  # Log evaluation event
                avg_policy = self._get_average_policy()

                metrics = {
                    "episode": episode,
                    "losses": [agent.loss for agent in self.agents],
                }

                if self.eval_callback:
                    eval_metrics = self.eval_callback(episode, avg_policy, metrics)
                    if eval_metrics:
                        for baseline, win_rate in eval_metrics.items():
                            if "win_rate" in baseline:
                                logger.info(f"Win rate {baseline}: {win_rate:.2%}")
                        logger.info(f"Eval metrics: {eval_metrics}")
                        metrics.update(eval_metrics)

                self.metrics_history.append(metrics)
            
            # Checkpoint
            if episode % self.config.checkpoint_freq == 0:
                self.save_checkpoint(f"checkpoint_ep_{episode}.pkl")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        final_metrics = {
            "total_time": total_time,
            "episodes": self.config.num_train_episodes,
            "final_losses": [agent.loss for agent in self.agents],
        }
        
        # Save final checkpoint
        self.save_checkpoint("final.pkl")
        
        return self._get_average_policy(), final_metrics
    
    def _get_average_policy(self) -> NFSPPolicies:
        """Get the current average policy from agents."""
        return NFSPPolicies(self.env, self.agents, nfsp.MODE.average_policy)
    
    def save_checkpoint(self, filename: str) -> Path:
        """Save training checkpoint.
        
        Note: NFSP checkpointing is limited - we save agent parameters
        but not replay buffers, so training cannot be fully resumed.
        
        Args:
            filename: Checkpoint filename.
            
        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = self.config.checkpoint_dir / filename
        
        checkpoint = {
            "episode": self.episode,
            "config": self.config.__dict__,
            "metrics_history": self.metrics_history,
            # Agent network parameters
            "agent_params": [
                {
                    "params_avg_network": agent.params_avg_network,
                    "params_q_network": agent._rl_agent.params_q_network,
                }
                for agent in self.agents
            ],
        }
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path
    
    def get_policy(self) -> NFSPPolicies:
        """Get current trained policy."""
        return self._get_average_policy()


def train_nfsp(
    config: Optional[NFSPConfig] = None,
    eval_callback: Optional[Callable] = None,
) -> Tuple[policy.Policy, Dict[str, Any]]:
    """Train NFSP agents on Truco.
    
    Args:
        config: Training configuration. Uses defaults if None.
        eval_callback: Optional evaluation callback.
        
    Returns:
        Trained average policy and training metrics.
    """
    if config is None:
        config = NFSPConfig()
    
    trainer = NFSPTrainer(config, eval_callback)
    return trainer.train()
