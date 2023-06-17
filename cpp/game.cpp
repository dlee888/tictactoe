#include "game.h"

FJML::Tensor create_game() { return FJML::Tensor::zeros({9}); }

bool is_game_over(const FJML::Tensor& game) {
    for (int i = 0; i < 3; i++) {
        if (game.at(i * 3) == game.at(i * 3 + 1) && game.at(i * 3 + 1) == game.at(i * 3 + 2) && game.at(i * 3) != 0) {
            return true;
        }
        if (game.at(i) == game.at(i + 3) && game.at(i + 3) == game.at(i + 6) && game.at(i) != 0) {
            return true;
        }
    }
    if (game.at(0) == game.at(4) && game.at(4) == game.at(8) && game.at(0) != 0) {
        return true;
    }
    if (game.at(2) == game.at(4) && game.at(4) == game.at(6) && game.at(2) != 0) {
        return true;
    }
    for (int i = 0; i < 9; i++) {
        if (game.at(i) == 0) {
            return false;
        }
    }
    return true;
}

std::tuple<FJML::Tensor, int, bool> step(const FJML::Tensor& game, int action) {
    FJML::Tensor new_game = game;
    if (new_game.at(action) != 0) {
        throw std::runtime_error("Invalid action");
    }
    new_game.at(action) = 1;
    new_game *= -1;
    if (is_game_over(new_game)) {
        return std::make_tuple(new_game, 1, true);
    }
    return std::make_tuple(new_game, 0, false);
}

std::vector<int> get_actions(const FJML::Tensor& game) {
    std::vector<int> actions;
    for (int i = 0; i < 9; i++) {
        if (game.at(i) == 0) {
            actions.push_back(i);
        }
    }
    return actions;
}

bool is_valid_action(const FJML::Tensor& game, int action) { return game.at(action) == 0; }
