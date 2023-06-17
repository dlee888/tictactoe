#include <FJML.h>
#include <iomanip>
#include <iostream>

#include "game.h"

void progress_bar(int curr, int tot, int bar_width = 69, double time_elapsed = -1);

int get_action(FJML::MLP& agent, const FJML::Tensor& game, float epsilon);

void print_game(const FJML::Tensor& game);
