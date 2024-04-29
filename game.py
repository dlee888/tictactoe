import typing

import torch


def create_game() -> torch.Tensor:
    """
    Returns a new game state.
    """
    return torch.zeros(9)


def is_game_over(game_state: torch.Tensor) -> int:
    """
    Returns 1 if the game is won, 0 if it's not over, 2 if tie.
    """
    for i in range(3):
        if game_state[i] == game_state[i + 3] == game_state[i + 6] != 0:
            return 1 
        if game_state[3 * i] == game_state[3 * i + 1] == game_state[3 * i + 2] != 0:
            return 1
    if game_state[0] == game_state[4] == game_state[8] != 0:
        return 1
    if game_state[2] == game_state[4] == game_state[6] != 0:
        return 1
    for i in range(9):
        if game_state[i] == 0:
            return 0
    return 2


def step(game_state: torch.Tensor, action: int) -> typing.Tuple[torch.Tensor, int, bool]:
    """
    Returns the next game state, the reward, and whether the game is over.
    """
    next_game_state = game_state.clone()
    if next_game_state[action] != 0:
        raise ValueError('Invalid action')
    next_game_state[action] = 1
    next_game_state = next_game_state * -1
    joever = is_game_over(next_game_state)
    return next_game_state, 1 if joever == 1 else 0, joever != 0


def get_actions(game_state: torch.Tensor) -> typing.List[int]:
    """
    Returns a list of valid actions.
    """
    return [i for i in range(9) if game_state[i] == 0]


def print_game_state(game_state: torch.Tensor) -> None:
    """
    Prints the game state.
    """
    for i in range(3):
        print(' '.join(['O' if game_state[3 * i + j] ==
              1 else 'X' if game_state[3 * i + j] == -1 else '.' for j in range(3)]))
    print()
