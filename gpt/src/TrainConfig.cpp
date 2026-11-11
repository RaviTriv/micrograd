#include "gpt/TrainConfig.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace micrograd::gpt {

namespace {

size_t parse_count(const std::string &flag, const std::string &value) {
  const bool digits =
      !value.empty() && std::all_of(value.begin(), value.end(), [](char c) {
        return c >= '0' && c <= '9';
      });
  if (!digits) {
    throw std::invalid_argument(flag + " needs a non-negative integer, got '" +
                                value + "'");
  }
  return static_cast<size_t>(std::stoull(value));
}

void require_positive(size_t value, const std::string &flag) {
  if (value == 0) {
    throw std::invalid_argument(flag + " must be positive");
  }
}

void validate(const TrainConfig &config) {
  require_positive(config.batch_size, "--batch-size");
  require_positive(config.grad_accum_steps, "--grad-accum-steps");
  require_positive(config.block_size, "--block-size");
  require_positive(config.n_layer, "--n-layer");
  require_positive(config.n_head, "--n-head");
  require_positive(config.n_embd, "--n-embd");
  require_positive(config.eval_iters, "--eval-iters");
  if (config.n_embd % config.n_head != 0) {
    throw std::invalid_argument("--n-embd must be divisible by --n-head");
  }
  if (config.muon_lr > 0.0f && config.learning_rate <= 0.0f) {
    throw std::invalid_argument("--muon-lr needs a positive --lr");
  }
}

}  // namespace

Device parse_device(const std::string &value) {
  if (value == "cpu") {
    return Device::CPU;
  }
  if (value == "cuda") {
    return Device::CUDA;
  }
  throw std::invalid_argument("Unknown device: " + value);
}

TrainConfig parse_args(int argc, char **argv, TrainConfig config) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto next_value = [&]() {
      if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + arg);
      }
      return std::string(argv[++i]);
    };

    if (arg == "--data") {
      config.data_path = next_value();
    } else if (arg == "--data-format") {
      config.data_format = next_value();
    } else if (arg == "--fineweb-dir") {
      config.fineweb_dir = next_value();
    } else if (arg == "--checkpoint") {
      config.checkpoint_path = next_value();
    } else if (arg == "--resume") {
      config.resume_path = next_value();
    } else if (arg == "--log") {
      config.log_path = next_value();
    } else if (arg == "--val-fraction") {
      config.val_fraction = std::stod(next_value());
    } else if (arg == "--batch-size") {
      config.batch_size = parse_count(arg, next_value());
    } else if (arg == "--grad-accum-steps") {
      config.grad_accum_steps = parse_count(arg, next_value());
    } else if (arg == "--block-size") {
      config.block_size = parse_count(arg, next_value());
    } else if (arg == "--n-layer") {
      config.n_layer = parse_count(arg, next_value());
    } else if (arg == "--n-head") {
      config.n_head = parse_count(arg, next_value());
    } else if (arg == "--n-embd") {
      config.n_embd = parse_count(arg, next_value());
    } else if (arg == "--dropout") {
      config.dropout = std::stof(next_value());
    } else if (arg == "--lr") {
      config.learning_rate = std::stof(next_value());
    } else if (arg == "--min-lr") {
      config.min_lr = std::stof(next_value());
    } else if (arg == "--weight-decay") {
      config.weight_decay = std::stof(next_value());
    } else if (arg == "--beta1") {
      config.beta1 = std::stof(next_value());
    } else if (arg == "--beta2") {
      config.beta2 = std::stof(next_value());
    } else if (arg == "--grad-clip") {
      config.grad_clip = std::stof(next_value());
    } else if (arg == "--warmup-iters") {
      config.warmup_iters = parse_count(arg, next_value());
    } else if (arg == "--max-iters") {
      config.max_iters = parse_count(arg, next_value());
    } else if (arg == "--eval-interval") {
      config.eval_interval = parse_count(arg, next_value());
    } else if (arg == "--eval-iters") {
      config.eval_iters = parse_count(arg, next_value());
    } else if (arg == "--checkpoint-interval") {
      config.checkpoint_interval = parse_count(arg, next_value());
    } else if (arg == "--throughput-interval") {
      config.throughput_interval = parse_count(arg, next_value());
    } else if (arg == "--grad-bucket-mb") {
      config.grad_bucket_mb = parse_count(arg, next_value());
    } else if (arg == "--bf16-grad-reduce") {
      config.bf16_grad_reduce = true;
    } else if (arg == "--device") {
      config.device = parse_device(next_value());
    } else if (arg == "--tf32") {
      config.tf32 = true;
    } else if (arg == "--bf16-matmul") {
      config.bf16_matmul = true;
    } else if (arg == "--bf16") {
      config.bf16 = true;
    } else if (arg == "--pad-vocab") {
      config.pad_vocab = true;
    } else if (arg == "--profile") {
      config.profile = true;
    } else if (arg == "--no-recompute") {
      config.recompute = false;
    } else if (arg == "--rotary") {
      config.arch.rotary = true;
    } else if (arg == "--qk-norm") {
      config.arch.qk_norm = true;
    } else if (arg == "--relu2") {
      config.arch.relu_squared = true;
    } else if (arg == "--muon-lr") {
      config.muon_lr = std::stof(next_value());
    } else if (arg == "--muon-fp32") {
      config.muon_fp32 = true;
    } else if (arg == "--seed") {
      config.seed = std::stoull(next_value());
    } else if (arg == "--sample") {
      config.sample = true;
    } else if (arg == "--prompt") {
      config.prompt = next_value();
    } else if (arg == "--max-new-tokens") {
      config.max_new_tokens = parse_count(arg, next_value());
    } else if (arg == "--temperature") {
      config.temperature = std::stof(next_value());
    } else if (arg == "--top-k") {
      config.top_k = parse_count(arg, next_value());
    } else if (arg == "--sample-delay-ms") {
      config.sample_delay_ms = parse_count(arg, next_value());
    } else {
      throw std::invalid_argument("Unknown flag: " + arg);
    }
  }

  validate(config);
  return config;
}

std::string format_flags(const TrainConfig &config) {
  std::ostringstream out;
  out << "--data " << config.data_path << " --data-format "
      << config.data_format << " --fineweb-dir " << config.fineweb_dir
      << " --checkpoint " << config.checkpoint_path;
  if (!config.resume_path.empty()) {
    out << " --resume " << config.resume_path;
  }
  out << " --log " << config.log_path << " --val-fraction "
      << config.val_fraction << " --batch-size " << config.batch_size
      << " --grad-accum-steps " << config.grad_accum_steps << " --block-size "
      << config.block_size << " --n-layer " << config.n_layer << " --n-head "
      << config.n_head << " --n-embd " << config.n_embd << " --dropout "
      << config.dropout << " --lr " << config.learning_rate << " --min-lr "
      << config.min_lr << " --weight-decay " << config.weight_decay
      << " --beta1 " << config.beta1 << " --beta2 " << config.beta2
      << " --grad-clip " << config.grad_clip << " --warmup-iters "
      << config.warmup_iters << " --max-iters " << config.max_iters
      << " --eval-interval " << config.eval_interval << " --eval-iters "
      << config.eval_iters << " --checkpoint-interval "
      << config.checkpoint_interval << " --throughput-interval "
      << config.throughput_interval << " --grad-bucket-mb "
      << config.grad_bucket_mb << " --device "
      << (config.device == Device::CUDA ? "cuda" : "cpu") << " --muon-lr "
      << config.muon_lr << " --seed " << config.seed;
  const std::pair<bool, const char *> switches[] = {
      {config.bf16_grad_reduce, "--bf16-grad-reduce"},
      {config.tf32, "--tf32"},
      {config.bf16_matmul, "--bf16-matmul"},
      {config.bf16, "--bf16"},
      {config.pad_vocab, "--pad-vocab"},
      {config.profile, "--profile"},
      {!config.recompute, "--no-recompute"},
      {config.arch.rotary, "--rotary"},
      {config.arch.qk_norm, "--qk-norm"},
      {config.arch.relu_squared, "--relu2"},
      {config.muon_fp32, "--muon-fp32"},
  };
  for (const auto &[enabled, flag] : switches) {
    if (enabled) {
      out << " " << flag;
    }
  }
  return out.str();
}

scalar_t LrAt(size_t it, const TrainConfig &config) {
  if (config.warmup_iters > 0 && it < config.warmup_iters) {
    return config.learning_rate * static_cast<scalar_t>(it + 1) /
           static_cast<scalar_t>(config.warmup_iters);
  }
  if (it >= config.max_iters) {
    return config.min_lr;
  }
  size_t denom = config.max_iters - config.warmup_iters;
  scalar_t progress = denom == 0
                          ? 1.0f
                          : static_cast<scalar_t>(it - config.warmup_iters) /
                                static_cast<scalar_t>(denom);
  scalar_t coeff =
      0.5f * (1.0f + std::cos(std::numbers::pi_v<scalar_t> * progress));
  return config.min_lr + (coeff * (config.learning_rate - config.min_lr));
}

void check_finite_loss(double loss, size_t step) {
  if (!std::isfinite(loss)) {
    throw std::runtime_error("training loss is not finite at step " +
                             std::to_string(step));
  }
}

}  // namespace micrograd::gpt
