import collections
import os
import random

import game
import model
import torch
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def get_qvals(agent: model.Model, game_state: torch.Tensor, actions: list) -> torch.Tensor:
    return torch.tensor([agent(torch.vstack((game_state, torch.nn.functional.one_hot(torch.tensor(action), 9)))) for action in actions])


def get_action(q_vals: torch.Tensor, actions: list[int], epsilon: float) -> int:
    if random.random() < epsilon:
        return random.choice(actions)
    return actions[torch.argmax(q_vals).item()]


def run_episode(agent: model.Model, replay_buffer: collections.deque, epsilon: float) -> None:
    game_state = game.create_game()
    done = False
    while not done:
        actions = game.get_actions(game_state)
        q_vals = get_qvals(agent, game_state, actions)
        action = get_action(q_vals, actions, epsilon)
        next_game_state, reward, done = game.step(game_state, action)
        replay_buffer.append(
            (game_state, action, reward, done))
        game_state = next_game_state


def learn(agent: model.Model, replay_batch: list, gamma: float, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """
    Train the agent using the given replay batch.
    """
    predictions = []
    targets = []
    for game_state, action, reward, done in replay_batch:
        prediction = agent(torch.vstack((game_state, torch.nn.functional.one_hot(torch.tensor(action), 9))))
        predictions.append(prediction)
        next_game_state, reward, done = game.step(game_state, action)
        if not done:
            next_actions = game.get_actions(next_game_state)
            next_q_vals = get_qvals(agent, next_game_state, next_actions)
            reward = reward - gamma * torch.max(next_q_vals).item()
        targets.append(reward)
    predictions = torch.stack(predictions).to(device).reshape((len(predictions),))
    targets = torch.tensor(targets, requires_grad=True).to(device).reshape((len(predictions),))
    # print(predictions, targets)
    loss = criterion(predictions, targets)
    loss.backward()
    # print(list(agent.parameters()))
    # print([x.grad for x in agent.parameters()])
    optimizer.step()


def train(agent: model.Model, *,
          criterion: torch.nn.Module = torch.nn.MSELoss(),
          optimizer: torch.optim.Optimizer,
          episodes: int = 1000, max_memory: int = 100000,
          epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
          gamma: float = 0.99,
          batch_size: int = 64, verbose: bool = False) -> None:
    """
    Train the agent using the given parameters.
    :param agent: The agent to train.
    :param criterion: The loss function to use.
    :param optimizer: The optimizer to use.
    :param episodes: The number of episodes to train for.
    :param max_memory: The maximum number of memories to store.
    :param epsilon: The initial epsilon value (for epsilon-greedy).
    :param epsilon_decay: The amount to decay epsilon by each episode.
    :param epsilon_min: The minimum value for epsilon.
    :param batch_size: The batch size to use for training.
    :param verbose: Whether to print out information about the training.
    """
    replay_buffer = collections.deque(maxlen=max_memory)
    for _ in range(5):
        run_episode(agent, replay_buffer, epsilon)
    for _ in range(episodes) if not verbose else tqdm.trange(episodes):
        run_episode(agent, replay_buffer, epsilon)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        replay_batch = random.sample(
            replay_buffer, min(batch_size, len(replay_buffer)))
        learn(agent, replay_batch, gamma, criterion, optimizer)
        torch.save(agent, 'model.pt')


if __name__ == '__main__':
    if 'model.pt' in os.listdir() and False:
        print('Loading model from model.pt')
        agent = torch.load('model.pt')
    else:
        agent = model.Model()
    optimizer = torch.optim.SGD(agent.parameters(), lr=0.05)
    train(agent, optimizer=optimizer, verbose=True,
          episodes=10, epsilon_decay=0.99, gamma=0.99, batch_size=64)
