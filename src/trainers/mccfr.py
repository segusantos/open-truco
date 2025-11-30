"""MCCFR (Monte Carlo CFR) trainer for Truco.

Uses OpenSpiel's C++ MCCFR implementations which are orders of magnitude
faster than Python-based Deep CFR for large games like Truco.

Available solvers:
- OutcomeSamplingMCCFRSolver: Samples single trajectories (fastest)
- ExternalSamplingMCCFRSolver: Samples all actions at player nodes
- CFRPlusSolver: CFR+ with regret matching+ (for smaller games)
"""

import pickle
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple
import logging

import pyspiel
from open_spiel.python import policy

from src.config import MCCFRConfig


logger = logging.getLogger(__name__)


class MCCFRTrainer:
    """Trainer for C++ MCCFR solvers on Truco."""
    
    SOLVER_TYPES = {
        "outcome": pyspiel.OutcomeSamplingMCCFRSolver,
        "external": pyspiel.ExternalSamplingMCCFRSolver,
    }
    
    def __init__(
        self,
        config: "MCCFRConfig",
        eval_callback: Optional[Callable] = None,
    ):
        """Initialize MCCFR trainer.
        
        Args:
            config: Training configuration.
            eval_callback: Optional callback for evaluation.
        """
        self.config = config
        self.eval_callback = eval_callback
        
        # Load game
        self.game = pyspiel.load_game(config.game_name)
        logger.info(f"Loaded game: {config.game_name}")
        logger.info(f"  Num players: {self.game.num_players()}")
        logger.info(f"  Num actions: {self.game.num_distinct_actions()}")
        
        # Create solver
        solver_class = self.SOLVER_TYPES.get(config.solver_type)
        if solver_class is None:
            raise ValueError(f"Unknown solver type: {config.solver_type}. "
                           f"Available: {list(self.SOLVER_TYPES.keys())}")
        
        if config.solver_type == "outcome":
            self.solver = solver_class(self.game, config.epsilon, config.seed)
        else:
            self.solver = solver_class(self.game, config.seed)
        
        logger.info(f"Created {config.solver_type} sampling MCCFR solver")
        
        # Training state
        self.iteration = 0
        self.metrics_history = []
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Tuple[policy.TabularPolicy, Dict[str, Any]]:
        """Run training loop.
        
        Returns:
            Trained average policy and metrics.
        """
        logger.info(f"Starting MCCFR training for {self.config.num_iterations:,} iterations")
        start_time = time.time()
        
        log_interval = max(1, self.config.num_iterations // 20)  # ~20 log messages
        
        for iteration in range(1, self.config.num_iterations + 1):
            self.iteration = iteration
            
            # Run one MCCFR iteration (C++ - very fast)
            self.solver.run_iteration()
            
            # Log progress
            if iteration % log_interval == 0 or iteration == self.config.num_iterations:
                elapsed = time.time() - start_time
                iters_per_sec = iteration / elapsed
                eta = (self.config.num_iterations - iteration) / iters_per_sec
                
                logger.info(
                    f"Iter {iteration:,}/{self.config.num_iterations:,} | "
                    f"Speed: {iters_per_sec:,.0f} it/s | "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"ETA: {eta:.1f}s"
                )
                
                self.metrics_history.append({
                    "iteration": iteration,
                    "elapsed": elapsed,
                    "iters_per_sec": iters_per_sec,
                })
            
            # Evaluate
            if iteration % self.config.eval_freq == 0 and self.eval_callback:
                avg_policy = self.solver.average_policy()
                eval_metrics = self.eval_callback(iteration, avg_policy, {})
                
                if eval_metrics:
                    logger.info(f"Eval: {eval_metrics}")
                    self.metrics_history[-1].update(eval_metrics)
            
            # Checkpoint
            if iteration % self.config.checkpoint_freq == 0:
                self.save_checkpoint(f"checkpoint_iter_{iteration}.pkl")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s "
                   f"({self.config.num_iterations / total_time:,.0f} it/s)")
        
        # Save final checkpoint
        self.save_checkpoint("final.pkl")
        
        final_metrics = {
            "total_time": total_time,
            "iterations": self.config.num_iterations,
            "iters_per_sec": self.config.num_iterations / total_time,
        }
        
        # Return the solver's policy (note: not serializable, use get_policy() for eval)
        return self.get_policy(), final_metrics
    
    def get_policy(self):
        """Get current average policy from the solver.
        
        Note: This returns a C++ policy object that cannot be pickled.
        Use it for evaluation during the same session.
        """
        return self.solver.average_policy()
    
    def save_checkpoint(self, filename: str) -> Path:
        """Save policy checkpoint.
        
        Note: C++ solver state cannot be fully serialized. We save the policy
        as a dictionary mapping info state strings to action probabilities.
        """
        checkpoint_path = self.config.checkpoint_dir / filename
        
        # Get the average policy and convert to a serializable format
        avg_policy = self.solver.average_policy()
        
        # The C++ policy object can't be pickled, so we just save metadata
        # The actual policy state is in the solver which we can't serialize
        checkpoint = {
            "iteration": self.iteration,
            "config": {
                "game_name": self.config.game_name,
                "solver_type": self.config.solver_type,
                "num_iterations": self.config.num_iterations,
                "epsilon": self.config.epsilon,
            },
            "metrics_history": self.metrics_history,
            # Note: Full policy serialization would require iterating all info states
            # which is expensive for large games. For now we just save iteration info.
        }
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path
    
    @staticmethod
    def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)


def train_mccfr(
    config: Optional["MCCFRConfig"] = None,
    eval_callback: Optional[Callable] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Train an MCCFR agent on Truco.
    
    Args:
        config: Training configuration.
        eval_callback: Optional evaluation callback.
        
    Returns:
        Trained policy and metrics.
    """
    if config is None:
        from src.config import MCCFRConfig
        config = MCCFRConfig()
    
    trainer = MCCFRTrainer(config, eval_callback)
    return trainer.train()
