#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gpt/Chat.h"
#include "gpt/Dataset.h"
#include "gpt/GPTConfig.h"
#include "gpt/Model.h"
#include "micrograd/Autograd.h"
#include "micrograd/NN.h"
#include "micrograd/Random.h"
#include "micrograd/Scalar.h"
#include "micrograd/Storage.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Optimizer.h"

#ifdef MICROGRAD_CUDA_ENABLED
#include "micrograd/backends/cuda/ops/Ops.h"
#endif

#ifndef CORPUS_DATA_PATH
#define CORPUS_DATA_PATH "data/input.txt"
#endif

namespace {

using micrograd::Backend;
using micrograd::cross_entropy;
using micrograd::Device;
using micrograd::global_rng;
using micrograd::load;
using micrograd::manual_seed;
using micrograd::NoGradGuard;
using micrograd::save;
using micrograd::scalar_t;
using micrograd::Storage;
using micrograd::Tensor;
using micrograd::gpt::Dataset;
using micrograd::gpt::gpt2_124m;
using micrograd::gpt::GPTConfig;
using micrograd::gpt::Model;
using micrograd::gpt::sample_token;

struct TrainConfig {
  std::string data_path = CORPUS_DATA_PATH;
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
  scalar_t dropout = 0.2f;
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
  Device device = Device::CPU;
  uint64_t seed = 1337;
  bool sample = false;
  std::string prompt = "\n";
  size_t max_new_tokens = 500;
  scalar_t temperature = 0.8f;
  size_t top_k = 200;
  size_t sample_delay_ms = 20;
};

Device parse_device(const std::string &value) {
  if (value == "cpu") {
    return Device::CPU;
  }
  if (value == "metal") {
    return Device::Metal;
  }
  if (value == "cuda") {
    return Device::CUDA;
  }
  throw std::invalid_argument("Unknown device: " + value);
}

TrainConfig parse_args(int argc, char **argv) {
  TrainConfig config;

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
    } else if (arg == "--checkpoint") {
      config.checkpoint_path = next_value();
    } else if (arg == "--resume") {
      config.resume_path = next_value();
    } else if (arg == "--log") {
      config.log_path = next_value();
    } else if (arg == "--val-fraction") {
      config.val_fraction = std::stod(next_value());
    } else if (arg == "--batch-size") {
      config.batch_size = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--grad-accum-steps") {
      config.grad_accum_steps = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--block-size") {
      config.block_size = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--n-layer") {
      config.n_layer = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--n-head") {
      config.n_head = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--n-embd") {
      config.n_embd = static_cast<size_t>(std::stoul(next_value()));
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
      config.warmup_iters = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--max-iters") {
      config.max_iters = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--eval-interval") {
      config.eval_interval = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--eval-iters") {
      config.eval_iters = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--checkpoint-interval") {
      config.checkpoint_interval =
          static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--device") {
      config.device = parse_device(next_value());
    } else if (arg == "--seed") {
      config.seed = std::stoull(next_value());
    } else if (arg == "--sample") {
      config.sample = true;
    } else if (arg == "--prompt") {
      config.prompt = next_value();
    } else if (arg == "--max-new-tokens") {
      config.max_new_tokens = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--temperature") {
      config.temperature = std::stof(next_value());
    } else if (arg == "--top-k") {
      config.top_k = static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--sample-delay-ms") {
      config.sample_delay_ms = static_cast<size_t>(std::stoul(next_value()));
    } else {
      throw std::invalid_argument("Unknown flag: " + arg);
    }
  }

  return config;
}

GPTConfig build_model_config(const TrainConfig &config) {
  GPTConfig model_config;
  model_config.depth = config.n_layer;
  model_config.n_layer = config.n_layer;
  model_config.n_embd = config.n_embd;
  model_config.n_head = config.n_head;
  model_config.n_kv_head = config.n_head;
  model_config.block_size = config.block_size;
  model_config.dropout = config.dropout;
  model_config.matrix_lr = 0.0f;
  model_config.embedding_lr = 0.0f;
  model_config.unembedding_lr = 0.0f;
  return model_config;
}

std::shared_ptr<Tensor> token_tensor(const std::vector<size_t> &tokens,
                                     size_t rows, size_t cols, Backend device) {
  std::vector<scalar_t> values(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++) {
    values[i] = static_cast<scalar_t>(tokens[i]);
  }
  auto tensor = std::make_shared<Tensor>(std::vector<size_t>{rows, cols},
                                         std::move(values));
  tensor->to(device);
  return tensor;
}

scalar_t GlobalGradNorm(
    const std::vector<std::shared_ptr<Tensor>> &parameters) {
  double sum_sq = 0.0;
  for (const auto &p : parameters) {
    sum_sq += static_cast<double>(micrograd::ops::GradNormSquared(*p));
  }
  return static_cast<scalar_t>(std::sqrt(sum_sq));
}

void ClipGradNorm(const std::vector<std::shared_ptr<Tensor>> &parameters,
                  scalar_t max_norm) {
  if (max_norm <= 0.0f) {
    return;
  }
  scalar_t norm = GlobalGradNorm(parameters);
  if (norm <= max_norm) {
    return;
  }
  scalar_t scale = max_norm / (norm + 1e-6f);
  for (const auto &p : parameters) {
    micrograd::ops::ScaleGrad(*p, scale);
  }
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

void write_u64(std::ostream &out, uint64_t value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

uint64_t read_u64(std::istream &in) {
  uint64_t value = 0;
  in.read(reinterpret_cast<char *>(&value), sizeof(value));
  return value;
}

class TrainOptimizer {
 public:
  TrainOptimizer(std::vector<std::shared_ptr<Tensor>> parameters,
                 scalar_t weight_decay, scalar_t beta1, scalar_t beta2,
                 scalar_t eps)
      : parameters_(std::move(parameters)),
        weight_decay_(weight_decay),
        beta1_(beta1),
        beta2_(beta2),
        eps_(eps) {
    for (const auto &p : parameters_) {
      decay_.push_back(p->shape().size() >= 2);
      m_.push_back(make_moment(*p));
      v_.push_back(make_moment(*p));
    }
  }

  void zero_grad() {
    for (auto &p : parameters_) {
      p->zero_grad();
    }
  }

  size_t step_count() const { return step_count_; }

  void write_state(std::ostream &out) const {
    write_u64(out, static_cast<uint64_t>(step_count_));
    for (size_t i = 0; i < parameters_.size(); i++) {
      write_tensor(out, *m_[i]);
      write_tensor(out, *v_[i]);
    }
  }

  void read_state(std::istream &in) {
    step_count_ = static_cast<size_t>(read_u64(in));
    for (size_t i = 0; i < parameters_.size(); i++) {
      read_tensor(in, m_[i]);
      read_tensor(in, v_[i]);
    }
  }

  void step(scalar_t learning_rate) {
    step_count_++;
    scalar_t bias_correction1 =
        1.0f - std::pow(beta1_, static_cast<scalar_t>(step_count_));
    scalar_t bias_correction2 =
        1.0f - std::pow(beta2_, static_cast<scalar_t>(step_count_));
#ifdef MICROGRAD_CUDA_ENABLED
    if (!parameters_.empty() &&
        parameters_.front()->backend() == Device::CUDA) {
      std::vector<micrograd::cuda::ops::AdamWTensor> tensors;
      tensors.reserve(parameters_.size());
      for (size_t i = 0; i < parameters_.size(); i++) {
        tensors.push_back(micrograd::cuda::ops::AdamWTensor{
            static_cast<scalar_t *>(
                parameters_[i]->data_storage().device_pointer()),
            static_cast<const scalar_t *>(
                parameters_[i]->grad_storage().device_pointer()),
            static_cast<scalar_t *>(m_[i]->data_storage().device_pointer()),
            static_cast<scalar_t *>(v_[i]->data_storage().device_pointer()),
            parameters_[i]->size(), decay_[i]});
      }
      micrograd::cuda::ops::FusedAdamWStep(tensors, learning_rate, beta1_,
                                           beta2_, eps_, weight_decay_,
                                           bias_correction1, bias_correction2);
      return;
    }
#endif
    for (size_t i = 0; i < parameters_.size(); i++) {
      micrograd::ops::AdamWStep(*parameters_[i], *m_[i], *v_[i], learning_rate,
                                beta1_, beta2_, eps_, weight_decay_, decay_[i],
                                bias_correction1, bias_correction2);
    }
  }

 private:
  static std::shared_ptr<Tensor> make_moment(const Tensor &param) {
    auto moment = std::make_shared<Tensor>(param.shape());
    moment->to(param.backend());
    return moment;
  }

  static void write_tensor(std::ostream &out, const Tensor &tensor) {
    Storage host = tensor.data_storage().copy_to(Device::CPU);
    out.write(static_cast<const char *>(host.data()),
              static_cast<std::streamsize>(host.bytes()));
  }

  static void read_tensor(std::istream &in, std::shared_ptr<Tensor> &tensor) {
    std::vector<scalar_t> values(tensor->size());
    in.read(reinterpret_cast<char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(scalar_t)));
    auto loaded = std::make_shared<Tensor>(tensor->shape(), std::move(values));
    loaded->to(tensor->backend());
    tensor = loaded;
  }

  std::vector<std::shared_ptr<Tensor>> parameters_;
  std::vector<bool> decay_;
  scalar_t weight_decay_;
  scalar_t beta1_;
  scalar_t beta2_;
  scalar_t eps_;
  size_t step_count_ = 0;
  std::vector<std::shared_ptr<Tensor>> m_;
  std::vector<std::shared_ptr<Tensor>> v_;
};

std::string optimizer_state_path(const std::string &checkpoint_path) {
  return checkpoint_path + ".opt";
}

void save_optimizer_state(const std::string &path,
                          const TrainOptimizer &optimizer) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for saving: " + path);
  }
  optimizer.write_state(file);
  if (!file) {
    throw std::runtime_error("Failed while writing optimizer state: " + path);
  }
}

void load_optimizer_state(const std::string &path, TrainOptimizer &optimizer) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for loading: " + path);
  }
  optimizer.read_state(file);
}

double estimate_loss(Model &model, const Dataset &dataset, Dataset::Split split,
                     const TrainConfig &config, std::mt19937_64 &rng) {
  const NoGradGuard no_grad;
  model.eval();

  double total = 0.0;
  for (size_t i = 0; i < config.eval_iters; i++) {
    Dataset::Batch batch =
        dataset.sample(config.batch_size, config.block_size, split, rng);
    auto input = token_tensor(batch.inputs, config.batch_size,
                              config.block_size, config.device);
    auto logits = model.forward(input);
    auto flat_logits = logits->reshape(
        {static_cast<int64_t>(config.batch_size * config.block_size),
         static_cast<int64_t>(dataset.vocab_size())});
    auto loss = cross_entropy(flat_logits, batch.targets);
    loss->to(Device::CPU);
    total += static_cast<double>(loss->at({0}));
  }

  model.train();
  return total / static_cast<double>(config.eval_iters);
}

void run_sample(const TrainConfig &config) {
  Dataset dataset(config.data_path, config.val_fraction);
  GPTConfig model_config = build_model_config(config);
  Model model(dataset.vocab_size(), model_config);

  std::vector<std::shared_ptr<Tensor>> parameters = model.parameters();
  for (auto &p : parameters) {
    p->to(config.device);
  }
  load(config.checkpoint_path, model);

  const NoGradGuard no_grad;
  model.eval();

  std::mt19937_64 rng(config.seed);
  std::vector<size_t> context = dataset.encode(config.prompt);
  if (context.empty()) {
    throw std::runtime_error("run_sample: prompt encodes to no tokens");
  }

  for (size_t token : context) {
    std::cout << dataset.decode(token);
  }
  std::cout.flush();

  for (size_t i = 0; i < config.max_new_tokens; i++) {
    size_t seq_len = std::min(context.size(), config.block_size);
    std::vector<size_t> window(context.end() - static_cast<int64_t>(seq_len),
                               context.end());
    auto input = token_tensor(window, 1, seq_len, config.device);
    auto logits = model.forward(input);
    logits->to(Device::CPU);

    std::vector<scalar_t> last_step(dataset.vocab_size());
    for (size_t v = 0; v < dataset.vocab_size(); v++) {
      last_step[v] = logits->at({0, seq_len - 1, v});
    }
    auto step_logits = std::make_shared<Tensor>(
        std::vector<size_t>{dataset.vocab_size()}, std::move(last_step));

    int32_t token =
        sample_token(step_logits, config.temperature, config.top_k, rng);
    context.push_back(static_cast<size_t>(token));

    std::cout << dataset.decode(static_cast<size_t>(token));
    std::cout.flush();
    std::this_thread::sleep_for(
        std::chrono::milliseconds(config.sample_delay_ms));
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    TrainConfig config = parse_args(argc, argv);
    manual_seed(config.seed);

    if (config.sample) {
      run_sample(config);
      return 0;
    }

    Dataset dataset(config.data_path, config.val_fraction);
    GPTConfig model_config = build_model_config(config);
    Model model(dataset.vocab_size(), model_config);

    std::vector<std::shared_ptr<Tensor>> parameters = model.parameters();
    for (auto &p : parameters) {
      p->to(config.device);
    }

    TrainOptimizer optimizer(parameters, config.weight_decay, config.beta1,
                             config.beta2, 1e-8f);

    size_t start_iter = 0;
    if (!config.resume_path.empty()) {
      load(config.resume_path, model);
      load_optimizer_state(optimizer_state_path(config.resume_path), optimizer);
      start_iter = optimizer.step_count();
    }

    std::ofstream log_file(config.log_path,
                           config.resume_path.empty()
                               ? std::ios::out
                               : std::ios::out | std::ios::app);
    if (!log_file.is_open()) {
      throw std::runtime_error("Could not open log file: " + config.log_path);
    }
    if (config.resume_path.empty()) {
      log_file << "step,train_loss,val_loss\n";
    }

    for (size_t it = start_iter; it < config.max_iters; it++) {
      if (config.eval_interval > 0 && it % config.eval_interval == 0) {
        double train_loss = estimate_loss(
            model, dataset, Dataset::Split::kTrain, config, global_rng());
        double val_loss = estimate_loss(model, dataset, Dataset::Split::kVal,
                                        config, global_rng());
        std::cout << "step " << it << ": train loss " << train_loss
                  << ", val loss " << val_loss << "\n";
        log_file << it << "," << train_loss << "," << val_loss << "\n";
        log_file.flush();
      }

      if (config.checkpoint_interval > 0 && it > 0 &&
          it % config.checkpoint_interval == 0) {
        save(config.checkpoint_path, model);
        save_optimizer_state(optimizer_state_path(config.checkpoint_path),
                             optimizer);
      }

      model.train();
      optimizer.zero_grad();
      for (size_t micro_step = 0; micro_step < config.grad_accum_steps;
           micro_step++) {
        Dataset::Batch batch =
            dataset.sample(config.batch_size, config.block_size,
                           Dataset::Split::kTrain, global_rng());
        auto input = token_tensor(batch.inputs, config.batch_size,
                                  config.block_size, config.device);
        auto logits = model.forward(input);
        auto flat_logits = logits->reshape(
            {static_cast<int64_t>(config.batch_size * config.block_size),
             static_cast<int64_t>(dataset.vocab_size())});
        auto loss = cross_entropy(flat_logits, batch.targets);
        auto scaled_loss =
            loss->div(static_cast<scalar_t>(config.grad_accum_steps));
        scaled_loss->backward();
      }
      ClipGradNorm(parameters, config.grad_clip);
      optimizer.step(LrAt(it, config));
    }

    double final_val_loss = estimate_loss(model, dataset, Dataset::Split::kVal,
                                          config, global_rng());
    std::cout << "final val loss " << final_val_loss << "\n";

    save(config.checkpoint_path, model);
    save_optimizer_state(optimizer_state_path(config.checkpoint_path),
                         optimizer);
    std::cout << "Saved trained model to " << config.checkpoint_path << "\n";
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
