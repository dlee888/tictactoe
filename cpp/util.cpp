#include "util.h"

void progress_bar(int curr, int tot, int bar_width, double time_elapsed) {
    float progress = (float)curr / tot;
    std::cout << "[";
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; i++) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %";
    if (time_elapsed > 0) {
        std::cout << std::fixed << std::setprecision(3) << " Time: " << time_elapsed
                  << ", ETA = " << time_elapsed * (1 - progress) / progress;
    }
    std::cout << "\r";
    std::cout.flush();
}

int get_action(FJML::MLP& agent, const FJML::Tensor& game, float epsilon) {
    std::vector<int> actions = get_actions(game);
    int action;
    if (rand() / (float)RAND_MAX < epsilon) {
        action = actions[rand() % actions.size()];
    } else {
        FJML::Tensor game_copy = game;
        game_copy.reshape({1, 9});
        FJML::Tensor q_values = agent.run(game_copy);
        action = actions[0];
        for (int i = 1; i < (int)actions.size(); i++) {
            if (q_values.at(0, actions[i]) > q_values.at(0, action)) {
                action = actions[i];
            }
        }
    }
    return action;
}

void print_game(const FJML::Tensor& game) {
    for (int i = 0; i < 9; i++) {
        if (i % 3 == 0) {
            std::cout << std::endl;
        }
        if (game.at(i) == 1) {
            std::cout << "O";
        } else if (game.at(i) == -1) {
            std::cout << "X";
        } else {
            std::cout << ".";
        }
    }
}
