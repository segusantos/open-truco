"""Deep CFR trainer for Truco.

Wraps OpenSpiel's Deep CFR implementation with training loop, 
checkpointing, evaluation, and logging for Truco experiments.
"""

import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
import logging

import numpy as np
import torch
import pyspiel

from open_spiel.python import policy
from open_spiel.python.pytorch import deep_cfr

from src.config import DeepCFRConfig


logger = logging.getLogger(__name__)


class DeepCFRTrainer:
    """Trainer wrapper for Deep CFR on Truco."""
    
    def __init__(
        self,
        config: DeepCFRConfig,
        eval_callback: Optional[Callable] = None,
    ):
        """Initialize Deep CFR trainer.
        
        Args:
            config: Training configuration.
            eval_callback: Optional callback for evaluation. Called with
                (iteration, policy, metrics) -> dict of eval metrics.
        """
        self.config = config
        self.eval_callback = eval_callback
        
        # Set device - Note: OpenSpiel's Deep CFR doesn't support GPU properly,
        # so we force CPU for now. The internal tensor creation doesn't move
        # tensors to GPU, causing device mismatches.
        if config.device == "cuda" and torch.cuda.is_available():
            logger.warning("OpenSpiel's Deep CFR has GPU compatibility issues. Forcing CPU.")
        self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Load game
        self.game = pyspiel.load_game(config.game_name)
        logger.info(f"Loaded game: {config.game_name}")
        logger.info(f"  Num players: {self.game.num_players()}")
        logger.info(f"  Num actions: {self.game.num_distinct_actions()}")
        logger.info(f"  Info state size: {self.game.information_state_tensor_shape()}")
        
        # Initialize solver
        self.solver = deep_cfr.DeepCFRSolver(
            game=self.game,
            policy_network_layers=config.policy_network_layers,
            advantage_network_layers=config.advantage_network_layers,
            num_iterations=1,  # We control iterations manually
            num_traversals=config.num_traversals,
            learning_rate=config.learning_rate,
            batch_size_advantage=config.batch_size_advantage,
            batch_size_strategy=config.batch_size_strategy,
            memory_capacity=config.memory_capacity,
        )
        
        # Training state
        self.iteration = 0
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Tuple[policy.Policy, Dict[str, Any]]:
        """Run full training loop.
        
        Returns:
            Trained policy and final metrics.
        """
        logger.info(f"Starting Deep CFR training for {self.config.num_iterations} iterations")
        start_time = time.time()
        
        all_advantage_losses: Dict[int, List[float]] = {
            p: [] for p in range(self.game.num_players())
        }
        
        for iteration in range(1, self.config.num_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()
            
            logger.info(f"Iteration {iteration}/{self.config.num_iterations} - running traversals...")
            
            # Run CFR traversals for each player
            for player in range(self.game.num_players()):
                logger.info(f"  Player {player}: {self.config.num_traversals} traversals...")
                for t in range(self.config.num_traversals):
                    self.solver._traverse_game_tree(
                        self.game.new_initial_state(), player
                    )
                    if (t + 1) % max(1, self.config.num_traversals // 5) == 0:
                        logger.info(f"    Traversal {t + 1}/{self.config.num_traversals}")
                
                # Train advantage network for this player
                logger.info(f"  Player {player}: training advantage network...")
                if self.solver._reinitialize_advantage_networks:
                    self.solver.reinitialize_advantage_network(player)
                
                adv_loss = self.solver._learn_advantage_network(player)
                if adv_loss is not None:
                    all_advantage_losses[player].append(float(adv_loss))
                    logger.info(f"  Player {player}: advantage loss = {adv_loss:.6f}")
            
            self.solver._iteration += 1
            
            iter_time = time.time() - iter_start
            
            # Log metrics
            if iteration % self.config.log_freq == 0:
                metrics = {
                    "iteration": iteration,
                    "time": iter_time,
                    "advantage_buffer_sizes": [
                        len(self.solver._advantage_memories[p])
                        for p in range(self.game.num_players())
                    ],
                    "strategy_buffer_size": len(self.solver._strategy_memories),
                }
                
                for p in range(self.game.num_players()):
                    if all_advantage_losses[p]:
                        metrics[f"adv_loss_p{p}"] = all_advantage_losses[p][-1]
                
                logger.info(
                    f"Iter {iteration}/{self.config.num_iterations} | "
                    f"Time: {iter_time:.2f}s | "
                    f"Adv buffers: {metrics['advantage_buffer_sizes']} | "
                    f"Strategy buffer: {metrics['strategy_buffer_size']}"
                )
                
                self.metrics_history.append(metrics)
            
            # Evaluate
            if iteration % self.config.eval_freq == 0 and self.eval_callback:
                # Train policy network for evaluation
                policy_loss = self.solver._learn_strategy_network()
                
                avg_policy = self._get_average_policy()
                eval_metrics = self.eval_callback(iteration, avg_policy, {
                    "policy_loss": float(policy_loss) if policy_loss else None,
                })
                
                if eval_metrics:
                    logger.info(f"Eval metrics: {eval_metrics}")
                    self.metrics_history[-1].update(eval_metrics)
            
            # Checkpoint
            if iteration % self.config.checkpoint_freq == 0:
                self.save_checkpoint(f"checkpoint_iter_{iteration}.pt")
        
        # Final policy training
        logger.info("Training final policy network...")
        policy_loss = self.solver._learn_strategy_network()
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        final_metrics = {
            "total_time": total_time,
            "iterations": self.config.num_iterations,
            "final_policy_loss": float(policy_loss) if policy_loss else None,
            "advantage_losses": {
                p: losses for p, losses in all_advantage_losses.items()
            },
        }
        
        # Save final checkpoint
        self.save_checkpoint("final.pt")
        
        return self._get_average_policy(), final_metrics
    
    def _get_average_policy(self) -> policy.Policy:
        """Get the current average policy from the solver."""
        return policy.tabular_policy_from_callable(
            self.game, self.solver.action_probabilities
        )
    
    def save_checkpoint(self, filename: str) -> Path:
        """Save training checkpoint.
        
        Args:
            filename: Checkpoint filename.
            
        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = self.config.checkpoint_dir / filename
        
        checkpoint = {
            "iteration": self.iteration,
            "config": self.config.__dict__,
            "policy_network_state": self.solver._policy_network.state_dict(),
            "advantage_network_states": [
                net.state_dict() for net in self.solver._advantage_networks
            ],
            "metrics_history": self.metrics_history,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.iteration = checkpoint["iteration"]
        self.metrics_history = checkpoint.get("metrics_history", [])
        
        self.solver._policy_network.load_state_dict(
            checkpoint["policy_network_state"]
        )
        
        for i, state_dict in enumerate(checkpoint["advantage_network_states"]):
            self.solver._advantage_networks[i].load_state_dict(state_dict)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (iteration {self.iteration})")
    
    def get_policy(self) -> policy.Policy:
        """Get current trained policy."""
        return self._get_average_policy()


def train_deep_cfr(
    config: Optional[DeepCFRConfig] = None,
    eval_callback: Optional[Callable] = None,
) -> Tuple[policy.Policy, Dict[str, Any]]:
    """Train a Deep CFR agent on Truco.
    
    Args:
        config: Training configuration. Uses defaults if None.
        eval_callback: Optional evaluation callback.
        
    Returns:
        Trained policy and training metrics.
    """
    if config is None:
        config = DeepCFRConfig()
    
    trainer = DeepCFRTrainer(config, eval_callback)
    return trainer.train()
