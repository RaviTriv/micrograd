#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "gpt/GPTConfig.h"
#include "micrograd/Device.h"
#include "micrograd/Scalar.h"

namespace micrograd::gpt {

struct TrainConfig {
  std::string data_path = "data/input.txt";
  std::string data_format = "char";
  std::string fineweb_dir = "fineweb-data";
  std::string checkpoint_path = "gpt.bin";
  std::string resume_path;
  std::string log_path = "training.log";
  double val_fraction = 0.1;
  size_t batch_size = 64;
  size_t grad_accum_steps = 1;
  size_t block_size = gpt2_124m().block_size;
  size_t n_layer = gpt2_124m().n_layer;
  size_t n_head = gpt2_124m().n_head;
  size_t n_embd = gpt2_124m().n_embd;
  scalar_t dropout = 0.0f;
  scalar_t learning_rate = 1e-3f;
  scalar_t min_lr = 1e-4f;
  scalar_t weight_decay = 0.1f;
  scalar_t beta1 = 0.9f;
  scalar_t beta2 = 0.99f;
  scalar_t grad_clip = 1.0f;
  size_t warmup_iters = 100;
  size_t max_iters = 5000;
  size_t eval_interval = 250;
  size_t eval_iters = 200;
  size_t checkpoint_interval = 250;
  size_t throughput_interval = 10;
  size_t grad_bucket_mb = 25;
  bool bf16_grad_reduce = false;
  Device device = Device::CPU;
  bool tf32 = false;
  bool bf16_matmul = false;
  bool bf16 = false;
  bool pad_vocab = false;
  bool profile = false;
  bool recompute = true;
  ArchOptions arch;
  scalar_t muon_lr = 0.0f;
  bool muon_fp32 = false;
  uint64_t seed = 1337;
  bool sample = false;
  std::string prompt = "\n";
  size_t max_new_tokens = 500;
  scalar_t temperature = 0.8f;
  size_t top_k = 200;
  size_t sample_delay_ms = 20;
};

Device parse_device(const std::string &value);

TrainConfig parse_args(int argc, char **argv, TrainConfig config = {});

std::string format_flags(const TrainConfig &config);

scalar_t LrAt(size_t it, const TrainConfig &config);

void check_finite_loss(double loss, size_t step);

}  // namespace micrograd::gpt
