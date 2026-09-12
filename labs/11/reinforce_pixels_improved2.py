#!/home/czechen/Projects/Deep_Learning/NPFL/bin/python3
import argparse

import gymnasium as gym
import numpy as np
import torch
import os

import npfl138
npfl138.require_version("2425.11")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")  # Increased from 10
parser.add_argument("--episodes", default=2000, type=int, help="Training episodes.")  # Increased from 1000
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")  # Reduced from 0.003
parser.add_argument("--model_path", default="model", type=str, help="Path to save/load the agent model.")
parser.add_argument("--entropy_weight", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--value_loss_weight", default=0.5, type=float, help="Value loss weight.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor.")

class Agent:
    # Use an accelerator if available.
    device = npfl138.trainable_module.get_auto_device()

    def __init__(self, env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace) -> None:
        self._args = args
        self._env = env  # Store env reference
        
        # Improved CNN architecture with batch normalization
        self._policy = torch.nn.Sequential(
            torch.nn.Conv2d(env.observation_space.shape[-1], 32, kernel_size=8, stride=4),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(1152, 256),  # Adjusted for new architecture
            torch.nn.ReLU(),
            torch.nn.Linear(256, env.action_space.n)
        ).to(self.device)
        
        # Value network with same feature extraction
        self._value_net = torch.nn.Sequential(
            torch.nn.Conv2d(env.observation_space.shape[-1], 32, kernel_size=8, stride=4),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(1152, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        ).to(self.device)

        # Use separate optimizers with learning rate scheduling
        self._policy_optimizer = torch.optim.Adam(self._policy.parameters(), lr=args.learning_rate)
        self._value_optimizer = torch.optim.Adam(self._value_net.parameters(), lr=args.learning_rate)
        
        # Learning rate schedulers
        self._policy_scheduler = torch.optim.lr_scheduler.StepLR(
            self._policy_optimizer, step_size=500, gamma=0.9
        )
        self._value_scheduler = torch.optim.lr_scheduler.StepLR(
            self._value_optimizer, step_size=500, gamma=0.9
        )
        
        # Track training statistics
        self.episode_returns = []
        self.policy_losses = []
        self.value_losses = []

    def save(self, path: str) -> None:
        """Saves the state of the policy and value networks."""
        self._policy.to("cpu")
        self._value_net.to("cpu")
        torch.save({
            'policy_state_dict': self._policy.state_dict(),
            'value_net_state_dict': self._value_net.state_dict(),
            'policy_optimizer_state_dict': self._policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self._value_optimizer.state_dict(),
        }, path)
        print(f"Agent model saved to {path}")
        self._policy.to(self.device)
        self._value_net.to(self.device)

    def load(self, path: str) -> None:
        """Loads the state of the policy and value networks."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self._policy.load_state_dict(checkpoint['policy_state_dict'])
        self._value_net.load_state_dict(checkpoint['value_net_state_dict'])
        if 'policy_optimizer_state_dict' in checkpoint:
            self._policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        if 'value_optimizer_state_dict' in checkpoint:
            self._value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self._policy.eval()
        self._value_net.eval()
        print(f"Agent model loaded from {path}")

    @npfl138.rl_utils.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        states = states.permute(0, 3, 1, 2)
        
        self._policy.train()
        self._value_net.train()
        
        # Forward pass
        policy_logits = self._policy(states)
        values = self._value_net(states).squeeze(-1)
        
        # Normalize returns for stability (but use original for value target)
        returns_mean = returns.mean()
        returns_std = returns.std()
        returns_normalized = (returns - returns_mean) / (returns_std + 1e-8)
        
        # Compute advantages
        advantages = returns_normalized - values.detach()
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss with entropy regularization
        log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
        probs = torch.nn.functional.softmax(policy_logits, dim=-1)
        
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(selected_log_probs * advantages).mean()
        
        # Entropy for exploration
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        policy_loss = policy_loss - self._args.entropy_weight * entropy
        
        # Value loss - use normalized returns as targets
        value_loss = torch.nn.functional.mse_loss(values, returns_normalized)
        
        # Update policy network
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), max_norm=0.5)
        self._policy_optimizer.step()
        
        # Update value network
        self._value_optimizer.zero_grad()
        (self._args.value_loss_weight * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(self._value_net.parameters(), max_norm=0.5)
        self._value_optimizer.step()
        
        # Store losses for monitoring
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        
        # Step schedulers
        self._policy_scheduler.step()
        self._value_scheduler.step()

    @npfl138.rl_utils.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._policy.eval()
        with torch.no_grad():
            if states.dim() == 3:
                states = states.unsqueeze(0)
            states = states.permute(0, 3, 1, 2)
            logits = self._policy(states)
            return torch.nn.functional.softmax(logits, dim=1)


def main(env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        agent = Agent(env, args)
        agent.load('model_new')

        # Final evaluation.
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                action = np.argmax(agent.predict(state))
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # Perform training
    agent = Agent(env, args)
    
    # Track best performance
    best_avg_return = -float('inf')
    recent_returns = []

    # Training
    for batch_idx in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        batch_episode_returns = []
        
        for _ in range(args.batch_size):
            # Perform an episode.
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            while not done:
                # Choose action according to probabilities
                action_probs = agent.predict(state)[0]
                action = np.random.choice(env.action_space.n, p=action_probs)  # Fixed hardcoded value

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # Compute discounted returns
            returns = []
            discounted_return = 0
            for reward in reversed(rewards):
                discounted_return = reward + args.gamma * discounted_return
                returns.insert(0, discounted_return)

            # Store episode return for monitoring
            batch_episode_returns.append(sum(rewards))

            # Append to training batch
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)

        # Train using the generated batch
        batch_actions = np.array(batch_actions)
        batch_states = np.array(batch_states)
        batch_returns = np.array(batch_returns)
        agent.train(batch_states, batch_actions, batch_returns)
        
        # Update tracking
        agent.episode_returns.extend(batch_episode_returns)
        recent_returns.extend(batch_episode_returns)
        recent_returns = recent_returns[-100:]  # Keep last 100 episodes
        
        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            avg_return = np.mean(recent_returns) if recent_returns else 0
            print(f"Batch {batch_idx}, Episodes {batch_idx * args.batch_size}-{(batch_idx + 1) * args.batch_size}")
            print(f"  Avg Return (last 100 eps): {avg_return:.2f}")
            print(f"  Avg Policy Loss: {np.mean(agent.policy_losses[-10:]):.4f}")
            print(f"  Avg Value Loss: {np.mean(agent.value_losses[-10:]):.4f}")
            
            # Save if best model
            if avg_return > best_avg_return and len(recent_returns) >= 100:
                best_avg_return = avg_return
                agent.save(args.model_path + "_best")
                print(f"  New best model saved! (avg return: {best_avg_return:.2f})")
    
    # Save final model
    agent.save(args.model_path)
    
    # Final evaluation
    print("\nFinal evaluation:")
    eval_returns = []
    for _ in range(10):
        state, done = env.reset(start_evaluation=True)[0], False
        episode_return = 0
        while not done:
            action = np.argmax(agent.predict(state))
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        eval_returns.append(episode_return)
    print(f"Average evaluation return: {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl138.rl_utils.EvaluationEnv(
        gym.make("npfl138/CartPolePixels-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)
