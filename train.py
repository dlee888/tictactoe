import collections
import os
import random

import torch
import tqdm

import game
import model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def get_action(agent: model.Model, game_state: torch.Tensor, epsilon: float) -> int:
    actions = game.get_actions(game_state)
    if random.random() < epsilon:
        return random.choice(actions)
    predictions = agent(game_state)
    best_action = actions[0]
    for action in actions:
        if predictions[action] > predictions[best_action]:
            best_action = action
    return best_action


def run_episode(agent: model.Model, replay_buffer: collections.deque, epsilon: float) -> None:
    game_state = game.create_game()
    done = False
    while not done:
        action = get_action(agent, game_state, epsilon)
        next_game_state, reward, done = game.step(game_state, action)
        replay_buffer.append(
            (game_state, action, reward, next_game_state, done))
        game_state = next_game_state


def learn(agent: model.Model, replay_batch: list, gamma: float, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """
    Train the agent using the given replay batch.
    """
    predictions = []
    targets = []
    for game_state, action, reward, next_game_state, done in replay_batch:
        prediction = agent(game_state)
        predictions.append(prediction)
        target = prediction.clone().detach()
        with torch.no_grad():
            actions = game.get_actions(next_game_state)
            for i in range(9):
                if i == action:
                    target[action] = reward
                    if not done:
                        target[action] += gamma * \
                            torch.max(agent(next_game_state))
                elif i not in actions:
                    target[i] = 0
        targets.append(target)
    predictions = torch.stack(predictions).to(device)
    targets = torch.stack(targets).to(device)
    # print(predictions, targets)
    loss = criterion(predictions, targets)
    loss.backward()
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
    for _ in range(episodes) if not verbose else tqdm.trange(episodes):
        run_episode(agent, replay_buffer, epsilon)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        replay_batch = random.sample(
            replay_buffer, min(batch_size, len(replay_buffer)))
        learn(agent, replay_batch, gamma, criterion, optimizer)
        torch.save(agent, 'model.pt')


if __name__ == '__main__':
    if 'model.pt' in os.listdir():
        print('Loading model from model.pt')
        agent = torch.load('model.pt')
    else:
        agent = model.Model()
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.005)
    train(agent, optimizer=optimizer, verbose=True, episodes=6969, gamma=0.99, batch_size=64)
