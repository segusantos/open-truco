"""PPO self-play trainer for Truco.

Implements PPO training where both players share the same policy
network, learning through self-play. This serves as a model-free
baseline for comparison with game-theoretic algorithms (Deep CFR, NFSP).
"""

import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
import logging

import numpy as np
import torch
from torch import nn, optim

import pyspiel
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.pytorch.ppo import PPOAgent, legal_actions_to_mask
from open_spiel.python.rl_agent import StepOutput

from src.config import PPOConfig


logger = logging.getLogger(__name__)


class TrucoVectorEnv:
    """Vectorized environment wrapper for Truco self-play.
    
    Runs multiple independent games in parallel, where both players
    use the same policy (self-play).
    """
    
    def __init__(self, game_name: str, num_envs: int, seed: int = 42):
        """Initialize vectorized environment.
        
        Args:
            game_name: Name of the game to load.
            num_envs: Number of parallel environments.
            seed: Random seed.
        """
        self.num_envs = num_envs
        self.game = pyspiel.load_game(game_name)
        self.num_players = self.game.num_players()
        
        # Get state/action specs
        state = self.game.new_initial_state()
        while state.is_chance_node():
            action = np.random.choice([a for a, _ in state.chance_outcomes()])
            state.apply_action(action)
        
        self.info_state_shape = tuple(self.game.information_state_tensor_shape())
        self.num_actions = self.game.num_distinct_actions()
        
        # Initialize environments
        np.random.seed(seed)
        self.states: List[pyspiel.State] = []
        self.current_players: List[int] = []
        self._reset_all()
    
    def _reset_all(self):
        """Reset all environments."""
        self.states = []
        self.current_players = []
        
        for _ in range(self.num_envs):
            state = self.game.new_initial_state()
            self._handle_chance(state)
            self.states.append(state)
            self.current_players.append(state.current_player())
    
    def _handle_chance(self, state: pyspiel.State):
        """Handle chance nodes by sampling."""
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np.random.choice(actions, p=probs)
            state.apply_action(action)
    
    def reset(self, env_idx: Optional[int] = None) -> Dict[str, Any]:
        """Reset environment(s).
        
        Args:
            env_idx: If provided, reset only this environment.
            
        Returns:
            Observations for all environments.
        """
        if env_idx is not None:
            state = self.game.new_initial_state()
            self._handle_chance(state)
            self.states[env_idx] = state
            self.current_players[env_idx] = state.current_player()
        else:
            self._reset_all()
        
        return self._get_obs()
    
    def _get_obs(self) -> Dict[str, Any]:
        """Get observations from all environments."""
        info_states = []
        legal_actions = []
        current_players = []
        
        for i, state in enumerate(self.states):
            player = state.current_player()
            info_states.append(state.information_state_tensor(player))
            legal_actions.append(state.legal_actions(player))
            current_players.append(player)
        
        return {
            "info_state": np.array(info_states),
            "legal_actions": legal_actions,
            "current_player": current_players,
        }
    
    def step(self, actions: List[int]) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[Dict]]:
        """Take actions in all environments.
        
        Args:
            actions: List of actions, one per environment.
            
        Returns:
            Tuple of (observations, rewards, dones, infos).
        """
        rewards = np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        
        for i, (state, action) in enumerate(zip(self.states, actions)):
            prev_player = state.current_player()
            state.apply_action(action)
            self._handle_chance(state)
            
            if state.is_terminal():
                # Get returns for the player who just acted
                returns = state.returns()
                rewards[i] = returns[prev_player]
                dones[i] = True
                infos[i]["episode_return"] = returns[prev_player]
                infos[i]["returns"] = returns
                
                # Reset this environment
                new_state = self.game.new_initial_state()
                self._handle_chance(new_state)
                self.states[i] = new_state
                self.current_players[i] = new_state.current_player()
            else:
                # Intermediate reward (could use shaped rewards here)
                rewards[i] = 0.0
                self.current_players[i] = state.current_player()
        
        return self._get_obs(), rewards, dones, infos


class PPOSelfPlayAgent(nn.Module):
    """PPO agent for Truco with MLP architecture."""
    
    def __init__(self, info_state_size: int, num_actions: int, hidden_sizes: Tuple[int, ...], device: str):
        super().__init__()
        
        self.num_actions = num_actions
        self.device = device
        
        # Build MLP
        layers = []
        in_size = info_state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        
        self.shared = nn.Sequential(*layers)
        self.actor = nn.Linear(in_size, num_actions)
        self.critic = nn.Linear(in_size, 1)
        
        # Initialize weights
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.constant_(self.critic.bias, 0)
        
        self.register_buffer("mask_value", torch.tensor(-1e9))
    
    def forward(self, x):
        return self.shared(x)
    
    def get_value(self, x):
        return self.critic(self.shared(x))
    
    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        hidden = self.shared(x)
        logits = self.actor(hidden)
        
        if legal_actions_mask is not None:
            logits = torch.where(legal_actions_mask, logits, self.mask_value)
        
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden), probs


class PPOTrainer:
    """PPO self-play trainer for Truco."""
    
    def __init__(
        self,
        config: PPOConfig,
        eval_callback: Optional[Callable] = None,
    ):
        """Initialize PPO trainer.
        
        Args:
            config: Training configuration.
            eval_callback: Optional callback for evaluation.
        """
        self.config = config
        self.eval_callback = eval_callback
        
        # Set device
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")
        
        # Set seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Create vectorized environment
        self.env = TrucoVectorEnv(config.game_name, config.num_envs, config.seed)
        self.game = self.env.game
        
        logger.info(f"Loaded game: {config.game_name}")
        logger.info(f"  Num players: {self.game.num_players()}")
        logger.info(f"  Num actions: {self.env.num_actions}")
        logger.info(f"  Info state shape: {self.env.info_state_shape}")
        logger.info(f"  Num parallel envs: {config.num_envs}")
        
        # Create agent
        info_state_size = int(np.prod(self.env.info_state_shape))
        self.agent = PPOSelfPlayAgent(
            info_state_size=info_state_size,
            num_actions=self.env.num_actions,
            hidden_sizes=config.hidden_layers_sizes,
            device=config.device,
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )
        
        # Training state
        self.total_steps = 0
        self.updates = 0
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Compute derived parameters
        self.batch_size = config.num_envs * config.num_steps
        self.minibatch_size = self.batch_size // config.num_minibatches
        self.num_updates = config.total_timesteps // self.batch_size
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Tuple[policy.Policy, Dict[str, Any]]:
        """Run full training loop.
        
        Returns:
            Trained policy and final metrics.
        """
        logger.info(f"Starting PPO self-play training for {self.config.total_timesteps} timesteps")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Minibatch size: {self.minibatch_size}")
        logger.info(f"  Num updates: {self.num_updates}")
        
        start_time = time.time()
        
        # Initialize buffers
        obs_buf = torch.zeros((self.config.num_steps, self.config.num_envs) + 
                              self.env.info_state_shape).to(self.device)
        actions_buf = torch.zeros((self.config.num_steps, self.config.num_envs), dtype=torch.long).to(self.device)
        logprobs_buf = torch.zeros((self.config.num_steps, self.config.num_envs)).to(self.device)
        rewards_buf = torch.zeros((self.config.num_steps, self.config.num_envs)).to(self.device)
        dones_buf = torch.zeros((self.config.num_steps, self.config.num_envs)).to(self.device)
        values_buf = torch.zeros((self.config.num_steps, self.config.num_envs)).to(self.device)
        legal_masks_buf = torch.zeros((self.config.num_steps, self.config.num_envs, self.env.num_actions), dtype=torch.bool).to(self.device)
        
        # Initialize environment
        obs_dict = self.env.reset()
        episode_returns = []
        
        for update in range(1, self.num_updates + 1):
            # Anneal learning rate
            if self.config.learning_rate > 0:
                frac = 1.0 - (update - 1) / self.num_updates
                lr = frac * self.config.learning_rate
                self.optimizer.param_groups[0]["lr"] = lr
            
            # Collect rollout
            for step in range(self.config.num_steps):
                self.total_steps += self.config.num_envs
                
                obs = torch.FloatTensor(obs_dict["info_state"]).to(self.device)
                legal_mask = legal_actions_to_mask(
                    obs_dict["legal_actions"], self.env.num_actions
                ).to(self.device)
                
                with torch.no_grad():
                    action, logprob, _, value, _ = self.agent.get_action_and_value(
                        obs, legal_actions_mask=legal_mask
                    )
                
                obs_buf[step] = obs
                actions_buf[step] = action
                logprobs_buf[step] = logprob
                values_buf[step] = value.flatten()
                legal_masks_buf[step] = legal_mask
                
                # Step environment
                obs_dict, rewards, dones, infos = self.env.step(action.cpu().numpy().tolist())
                
                rewards_buf[step] = torch.FloatTensor(rewards).to(self.device)
                dones_buf[step] = torch.FloatTensor(dones).to(self.device)
                
                # Track episode returns
                for info in infos:
                    if "episode_return" in info:
                        episode_returns.append(info["episode_return"])
            
            # Compute advantages using GAE
            with torch.no_grad():
                next_obs = torch.FloatTensor(obs_dict["info_state"]).to(self.device)
                next_value = self.agent.get_value(next_obs).flatten()
                
                advantages = torch.zeros_like(rewards_buf)
                lastgaelam = 0
                for t in reversed(range(self.config.num_steps)):
                    if t == self.config.num_steps - 1:
                        next_values = next_value
                    else:
                        next_values = values_buf[t + 1]
                    
                    next_nonterminal = 1.0 - dones_buf[t]
                    delta = rewards_buf[t] + self.config.gamma * next_values * next_nonterminal - values_buf[t]
                    advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * next_nonterminal * lastgaelam
                
                returns = advantages + values_buf
            
            # Flatten batches
            b_obs = obs_buf.reshape((-1,) + self.env.info_state_shape)
            b_actions = actions_buf.reshape(-1)
            b_logprobs = logprobs_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buf.reshape(-1)
            b_legal_masks = legal_masks_buf.reshape(-1, self.env.num_actions)
            
            # Optimize policy and value networks
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            
            for _ in range(self.config.update_epochs):
                np.random.shuffle(b_inds)
                
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    _, newlogprob, entropy, newvalue, _ = self.agent.get_action_and_value(
                        b_obs[mb_inds],
                        legal_actions_mask=b_legal_masks[mb_inds],
                        action=b_actions[mb_inds],
                    )
                    
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item())
                    
                    mb_advantages = b_advantages[mb_inds]
                    if self.config.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.config.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -self.config.clip_coef, self.config.clip_coef
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    
                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
            
            self.updates = update
            
            # Log metrics
            if update % (self.config.log_freq // self.batch_size + 1) == 0 or update == 1:
                sps = int(self.total_steps / (time.time() - start_time))
                avg_return = np.mean(episode_returns[-100:]) if episode_returns else 0.0
                
                metrics = {
                    "update": update,
                    "total_steps": self.total_steps,
                    "sps": sps,
                    "avg_return": avg_return,
                    "pg_loss": pg_loss.item(),
                    "v_loss": v_loss.item(),
                    "entropy": entropy_loss.item(),
                    "approx_kl": approx_kl.item(),
                    "clipfrac": np.mean(clipfracs),
                }
                
                logger.info(
                    f"Update {update}/{self.num_updates} | "
                    f"Steps: {self.total_steps} | "
                    f"SPS: {sps} | "
                    f"Avg Return: {avg_return:.3f} | "
                    f"PG Loss: {pg_loss.item():.4f} | "
                    f"V Loss: {v_loss.item():.4f}"
                )
                
                self.metrics_history.append(metrics)
            
            # Evaluate
            if update % (self.config.eval_freq // self.batch_size + 1) == 0 and self.eval_callback:
                ppo_policy = self._get_policy()
                eval_metrics = self.eval_callback(update, ppo_policy, metrics)
                if eval_metrics:
                    logger.info(f"Eval metrics: {eval_metrics}")
            
            # Checkpoint
            if update % (self.config.checkpoint_freq // self.batch_size + 1) == 0:
                self.save_checkpoint(f"checkpoint_update_{update}.pt")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        final_metrics = {
            "total_time": total_time,
            "total_steps": self.total_steps,
            "updates": self.updates,
            "final_avg_return": np.mean(episode_returns[-100:]) if episode_returns else 0.0,
        }
        
        # Save final checkpoint
        self.save_checkpoint("final.pt")
        
        return self._get_policy(), final_metrics
    
    def _get_policy(self) -> "PPOPolicy":
        """Get the current policy as a callable."""
        return PPOPolicy(self.game, self.agent, self.device)
    
    def save_checkpoint(self, filename: str) -> Path:
        """Save training checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / filename
        
        checkpoint = {
            "total_steps": self.total_steps,
            "updates": self.updates,
            "config": self.config.__dict__,
            "agent_state": self.agent.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.total_steps = checkpoint["total_steps"]
        self.updates = checkpoint["updates"]
        self.metrics_history = checkpoint.get("metrics_history", [])
        
        self.agent.load_state_dict(checkpoint["agent_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (update {self.updates})")
    
    def get_policy(self) -> "PPOPolicy":
        """Get current trained policy."""
        return self._get_policy()


class PPOPolicy(policy.Policy):
    """Policy wrapper for trained PPO agent."""
    
    def __init__(self, game, agent: PPOSelfPlayAgent, device):
        super().__init__(game, list(range(game.num_players())))
        self.agent = agent
        self.device = device
        self.agent.eval()
    
    def action_probabilities(self, state, player_id=None):
        """Returns action probabilities for the given state."""
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        
        info_state = torch.FloatTensor(
            state.information_state_tensor(cur_player)
        ).unsqueeze(0).to(self.device)
        
        legal_mask = torch.zeros(1, self.agent.num_actions, dtype=torch.bool).to(self.device)
        legal_mask[0, legal_actions] = True
        
        with torch.no_grad():
            _, _, _, _, probs = self.agent.get_action_and_value(
                info_state, legal_actions_mask=legal_mask
            )
        
        probs = probs[0].cpu().numpy()
        return {action: float(probs[action]) for action in legal_actions}


def train_ppo(
    config: Optional[PPOConfig] = None,
    eval_callback: Optional[Callable] = None,
) -> Tuple[policy.Policy, Dict[str, Any]]:
    """Train a PPO agent on Truco via self-play.
    
    Args:
        config: Training configuration. Uses defaults if None.
        eval_callback: Optional evaluation callback.
        
    Returns:
        Trained policy and training metrics.
    """
    if config is None:
        config = PPOConfig()
    
    trainer = PPOTrainer(config, eval_callback)
    return trainer.train()
