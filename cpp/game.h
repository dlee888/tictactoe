#include <tuple>
#include <vector>

#include <FJML.h>

FJML::Tensor create_game();

bool is_game_over(const FJML::Tensor& game);

std::tuple<FJML::Tensor, int, bool> step(const FJML::Tensor& game, int action);

std::vector<int> get_actions(const FJML::Tensor& game);

bool is_valid_action(const FJML::Tensor& game, int action);
