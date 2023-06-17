#include <chrono>
#include <deque>
#include <iomanip>
#include <iostream>
#include <tuple>

#include <FJML.h>

#include "game.h"
#include "util.h"

void run_episode(FJML::MLP& agent, std::deque<std::tuple<FJML::Tensor, int, int, FJML::Tensor, bool>>& memory,
                 float epsilon) {
    FJML::Tensor game = create_game();
    bool done = false;
    while (!done) {
        std::vector<int> actions = get_actions(game);
        int action = get_action(agent, game, epsilon);
        std::tuple<FJML::Tensor, int, bool> step_result = step(game, action);
        FJML::Tensor new_game = std::get<0>(step_result);
        int new_reward = std::get<1>(step_result);
        bool new_done = std::get<2>(step_result);
        memory.push_back(std::make_tuple(game, action, new_reward, new_game, new_done));
        game = std::move(new_game);
        done = new_done;
    }
}

#define time_elapsed                                                                                                   \
    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() /     \
        1000.0

void train(FJML::MLP& agent, std::deque<std::tuple<FJML::Tensor, int, int, FJML::Tensor, bool>>& memory, float gamma,
           int batch_size) {
    std::vector<FJML::Tensor> x_batch;
    std::vector<FJML::Tensor> y_batch;
    for (int i = 0; i < batch_size; i++) {
        int index = rand() % memory.size();
        std::tuple<FJML::Tensor, int, int, FJML::Tensor, bool>& sample = memory[index];
        FJML::Tensor game = std::get<0>(sample);
        int action = std::get<1>(sample);
        int reward = std::get<2>(sample);
        FJML::Tensor new_game = std::get<3>(sample);
        bool done = std::get<4>(sample);
        game.reshape({1, 9});
        new_game.reshape({1, 9});
        FJML::Tensor q_values = agent.run(game);
        q_values.reshape({9});
        FJML::Tensor new_q_values = agent.run(new_game);
        float target = reward;
        if (!done) {
            target += gamma * FJML::LinAlg::max(new_q_values);
        }
        q_values.at(action) = target;
        game.reshape({9});
        x_batch.push_back(game);
        y_batch.push_back(q_values);
    }
    agent.grad_descent(FJML::Tensor::array(x_batch), FJML::Tensor::array(y_batch));
}

void run_training(FJML::MLP& agent, int episodes, int max_memory = 100000, float epsilon = 1.0,
                  float epsilon_decay = 0.995, float epsilon_min = 0.01, float gamma = 0.99, int batch_size = 64) {
    std::deque<std::tuple<FJML::Tensor, int, int, FJML::Tensor, bool>> memory;
    auto start_time = std::chrono::system_clock::now();
    for (int episode = 0; episode < episodes; episode++) {
        run_episode(agent, memory, epsilon);
        while ((int)memory.size() > max_memory) {
            memory.pop_front();
        }
        epsilon = std::max(epsilon * epsilon_decay, epsilon_min);
        train(agent, memory, gamma, batch_size);
        progress_bar(episode + 1, episodes, 69, time_elapsed);
        agent.save("model.fjml");
    }
    std::cout << std::endl;
}

int main() {
    srand(time(NULL));
    FJML::MLP agent({new FJML::Layers::Dense(9, 32, FJML::Activations::relu),
                     new FJML::Layers::Dense(32, 64, FJML::Activations::relu),
                     new FJML::Layers::Dense(64, 9, FJML::Activations::sigmoid)},
                    FJML::Loss::mse, new FJML::Optimizers::Adam(0.005));
    run_training(agent, 6969);
}
