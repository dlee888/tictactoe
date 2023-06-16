import torch

import game
import train

if __name__ == '__main__':
    print('Loading model from model.pt')
    agent = torch.load('model.pt')
    print('Playing against the agent. You are O, the agent is X.')
    state = game.create_game()
    while not game.is_game_over(state):
        game.print_game_state(state)
        print('Your turn. Enter a number from 0-8.')
        action = int(input())
        state, reward, done = game.step(state, action)
        print(state)
        if done:
            break
        print('Agent\'s turn.')
        print(agent(state))
        action = train.get_action(agent, state, 0)
        state, reward, done = game.step(state, action)
        print(action, state)
