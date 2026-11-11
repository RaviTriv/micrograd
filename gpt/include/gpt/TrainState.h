#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "gpt/SampleOrder.h"
#include "gpt/TrainOptimizer.h"

namespace micrograd::gpt {

std::string optimizer_state_path(const std::string &checkpoint_path);
void save_optimizer_state(const std::string &path,
                          const TrainOptimizer &optimizer);
void load_optimizer_state(const std::string &path, TrainOptimizer &optimizer,
                          std::optional<size_t> expected_step = std::nullopt);

std::string data_state_path(const std::string &checkpoint_path);
void save_data_state(const std::string &path, const SampleOrder &order,
                     size_t step = 0);
void load_data_state(const std::string &path, size_t samples_from_steps,
                     SampleOrder &order,
                     std::optional<size_t> expected_step = std::nullopt);

std::string rng_state_path(const std::string &checkpoint_path);
void save_rng_state(const std::string &path, size_t step = 0);
void load_rng_state(const std::string &path,
                    std::optional<size_t> expected_step = std::nullopt);

size_t check_train_state(const std::string &checkpoint_path);

std::string checkpoint_set_path(const std::string &checkpoint_path,
                                size_t step);
void write_latest_checkpoint(const std::string &checkpoint_path, size_t step);
std::optional<size_t> read_latest_checkpoint(
    const std::string &checkpoint_path);
std::string resolve_resume_path(const std::string &checkpoint_path);
void remove_checkpoint_set(const std::string &set_path);

}  // namespace micrograd::gpt
