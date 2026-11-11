#pragma once

#include <cstddef>
#include <string>

#include "gpt/SampleOrder.h"
#include "gpt/TrainOptimizer.h"

namespace micrograd::gpt {

std::string optimizer_state_path(const std::string &checkpoint_path);
void save_optimizer_state(const std::string &path,
                          const TrainOptimizer &optimizer);
void load_optimizer_state(const std::string &path, TrainOptimizer &optimizer);

std::string data_state_path(const std::string &checkpoint_path);
void save_data_state(const std::string &path, const SampleOrder &order);
void load_data_state(const std::string &path, size_t samples_from_steps,
                     SampleOrder &order);

std::string rng_state_path(const std::string &checkpoint_path);
void save_rng_state(const std::string &path);
void load_rng_state(const std::string &path);

}  // namespace micrograd::gpt
