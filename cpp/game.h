#include <tuple>
#include <vector>

#include <FJML.h>

FJML::Tensor create_game();

/**
 * Returns 1 if a player has won, 0 if the game is still going, and -1 if the game is a draw.
 */
int is_game_over(const FJML::Tensor& game);

std::tuple<FJML::Tensor, float, bool> step(const FJML::Tensor& game, int action);

std::vector<int> get_actions(const FJML::Tensor& game);

bool is_valid_action(const FJML::Tensor& game, int action);
