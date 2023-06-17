#include <FJML.h>

#include "game.h"
#include "util.h"

int main() {
    std::cout << "Loading model from model.fjml" << std::endl;
    FJML::MLP model;
    model.load("model.fjml");
    std::cout << "Playing against the agent. You are O, the agent is X." << std::endl;
    FJML::Tensor game = create_game();
    bool done = false;
    float reward;
    while (!done) {
        print_game(game);
        std::cout << std::endl;
        std::cout << "Enter your move (0-8): ";
        int action;
        std::cin >> action;
        while (!is_valid_action(game, action)) {
            std::cout << "Invalid action. Enter your move (0-8): ";
            std::cin >> action;
        }
        std::tie(game, reward, done) = step(game, action);
        if (done) {
            if (reward == 0.5) {
                std::cout << "Draw!" << std::endl;
                return 0;
            }
            std::cout << "You win!" << std::endl;
            return 0;
        }
        std::cout << "Agent's turn..." << std::endl;
        game.reshape({1, 9});
        std::cout << model.run(game) << std::endl;
        game.reshape({9});
        action = get_action(model, game, 0);
        std::tie(game, reward, done) = step(game, action);
        if (done) {
            if (reward == 0.5) {
                std::cout << "Draw!" << std::endl;
                return 0;
            }
            std::cout << "You lose!" << std::endl;
            return 0;
        }
    }
}
