#!/home/czechen/Projects/Deep_Learning/NPFL/bin/python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import npfl138
npfl138.require_version("2425.11")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")


class ActorCritic(torch.nn.Module):
    def __init__(self, observation_shape: tuple, num_actions: int):
        super().__init__()
        in_channels = observation_shape[2]

        # Shared CNN feature extractor (backbone)
        # This architecture is inspired by standard models used in deep RL (like Nature DQN)
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=4, stride=4), # (B, 16, 19, 19) for 80x80 input
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=2, stride=2),          # (B, 32, 8, 8)
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        # To connect the backbone to the heads, we need to know the size of the flattened features.
        # We can compute this automatically by doing a forward pass with a dummy tensor.
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, observation_shape[0], observation_shape[1])
            flattened_size = self.backbone(dummy_input).shape[1]

        # Policy head (Actor)
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(flattened_size, 64),
            torch.nn.ReLU(),
            #torch.nn.Dropout(0.5),
            torch.nn.Linear(64, num_actions)
        )

        # Value head (Critic)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(flattened_size, 128),
            torch.nn.ReLU(),
            #torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 1)
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.
        Args:
            states: A tensor of shape (B, C, H, W).
        Returns:
            A tuple containing (action_logits, state_values).
        """
        features = self.backbone(states)
        action_logits = self.policy_head(features)
        state_values = self.value_head(features)
        return action_logits, state_values.squeeze(-1)

class Agent:
    # Use an accelerator if available.
    device = npfl138.trainable_module.get_auto_device()

    def __init__(self, env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model of the policy. Note that the shape
        # of the observations is available in `env.observation_space.shape`
        # and the number of actions in `env.action_space.n`.
        #
        # Apart from the policy network defined in `reinforce` assignment, you
        # also need a value network for computing the baseline, returning
        # a single output with no activation.
        #
        # Using Adam optimizer with the given `args.learning_rate` for both models
        # is a good default.
        self._args = args
        
        self._model = ActorCritic(env.observation_space.shape, env.action_space.n).to(self.device)


        self._optimizer = torch.optim.Adam(self._model.parameters(), args.learning_rate)
        self._ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self._mse_loss_fn = torch.nn.MSELoss()

    def save(self, path: str) -> None:
        # Save the state_dict of the single model
        torch.save(self._model.state_dict(), path)
        print(f"Agent model saved to {path}")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        # Load the state_dict into the single model
        self._model.load_state_dict(torch.load(path, map_location=self.device))
        self._model.eval()
        print(f"Agent model loaded from {path}")

    # The `npfl138.rl_utils.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
 
    @npfl138.rl_utils.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        self._model.train()
        states = states.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)

        # --- REFACTORED: Single forward pass ---
        policy_logits, value_pred = self._model(states)

        # Value loss (critic loss)
        value_loss = self._mse_loss_fn(value_pred, returns)

        # Policy loss (actor loss)
        advantage = (returns - value_pred).detach() # .detach() is CRITICAL
        policy_loss = self._ce_loss_fn(policy_logits, actions)
        actor_loss = torch.mean(policy_loss * advantage)

        # Total loss
        # A common practice is to scale the value loss. 0.5 is a standard coefficient.
        loss = actor_loss + value_loss

        self._optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=0.5)
        self._optimizer.step()

    @npfl138.rl_utils.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        states = states.unsqueeze(0).permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        
        # We only need the policy logits for prediction
        policy_logits, _ = self._model(states)
        return torch.nn.Softmax(dim=1)(policy_logits)


def main(env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    agent = Agent(env, args)
    if args.recodex:
        # TODO: Load the agent.
        agent.load(args.model_path)
        # Final evaluation.
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                # TODO: Choose a greedy action.
                action = np.argmax(agent.predict(state))
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform an episode.
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            while not done:
                # TODO(reinforce): Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                action_probs = agent.predict(state)[0]
                action = np.random.choice(env.action_space.n, p=action_probs)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO(reinforce): Compute returns by summing rewards.
            returns = np.cumsum(rewards[::-1])[::-1]

            # TODO(reinforce): Append states, actions and returns to the training batch.
            batch_states.append(states)
            batch_returns.append(returns)
            batch_actions.append(actions)

        # TODO(reinforce): Train using the generated batch.
        batch_actions = np.concatenate(batch_actions)
        batch_states = np.concatenate(batch_states)
        batch_returns = np.concatenate(batch_returns)
        agent.train(batch_states,batch_actions,batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO(reinforce): Choose a greedy action.
            action = np.argmax(agent.predict(state))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated



if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl138.rl_utils.EvaluationEnv(
        gym.make("npfl138/CartPolePixels-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)
