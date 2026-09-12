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
parser.add_argument("--batch_size", default=3, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=32, type=int, help="Size of hidden layer.")
parser.add_argument("--value_layer_size", default=128, type=int, help="Size of hidden layer in Value network.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")


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
        self._policy = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0],args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size,env.action_space.n),
        ).to(self.device)
        
        self._value = torch.nn.Sequential(
                torch.nn.Linear(env.observation_space.shape[0],args.value_layer_size),
                torch.nn.ReLU(),
                torch.nn.Linear(args.value_layer_size,1))

        
        params = list(self._policy.parameters()) + list(self._value.parameters())
        self._policy_optimizer = torch.optim.Adam(params,args.learning_rate)
        self._value_optimizer = torch.optim.Adam(self._value.parameters(),args.learning_rate)
        self._CEloss = torch.nn.CrossEntropyLoss(reduction='none')
        self._MSE = torch.nn.MSELoss()

    # The `npfl138.rl_utils.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl138.rl_utils.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Perform training.
        # You should:
        # - compute the predicted baseline using the baseline model,
        # - train the policy model, using `returns` - `predicted_baseline` as
        #   advantage estimate,
        # - train the baseline model to predict `returns`.
        #
        # Note that predicting returns in 0-500 range is challenging for the network, given
        # that the default initialization tries to keep variance -- it might be helpful for
        # the network if you predict returns in a smaller range.
        policy_pred = self._policy(states)
        CE_Loss = self._CEloss(policy_pred,actions)
        
        value_pred = self._value(states).squeeze(1)
        loss2 = self._MSE(value_pred,returns)

        loss1 = torch.sum(CE_Loss*(returns-value_pred))
        loss = loss1 + loss2 
        self._policy_optimizer.zero_grad()
        #self._value_optimizer.zero_grad()
        loss.backward()
        #loss1.backward(retain_graph=True)
        #loss2.backward(retain_graph=True)
        self._policy_optimizer.step()
        #self._value_optimizer.step()

    @npfl138.rl_utils.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        # TODO(reinforce): Define the prediction method returning policy probabilities.
        return torch.nn.Softmax(dim=0)(self._policy(states))


def main(env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Construct the agent.
    agent = Agent(env, args)

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
                actions_probs = agent.predict(state)
                action = np.random.choice(2,1,p=actions_probs)[0]

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
    main_env = npfl138.rl_utils.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)
