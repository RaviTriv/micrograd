#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gpt/Chat.h"
#include "gpt/Dataset.h"
#include "gpt/GPTConfig.h"
#include "gpt/Model.h"
#include "micrograd/Autograd.h"
#include "micrograd/Distributed.h"
#include "micrograd/NN.h"
#include "micrograd/Profiler.h"
#include "micrograd/Random.h"
#include "micrograd/Scalar.h"
#include "micrograd/Storage.h"
#include "micrograd/Tensor.h"
#include "micrograd/ops/Optimizer.h"

#ifdef MICROGRAD_CUDA_ENABLED
#include "micrograd/backends/cuda/CudaContext.h"
#include "micrograd/backends/cuda/ops/Ops.h"
#endif

#ifndef CORPUS_DATA_PATH
#define CORPUS_DATA_PATH "data/input.txt"
#endif

#ifndef FINEWEB_DATA_DIR
#define FINEWEB_DATA_DIR "fineweb-data"
#endif

namespace {

using micrograd::Backend;
using micrograd::cross_entropy;
using micrograd::Device;
namespace distributed = micrograd::distributed;
using micrograd::global_rng;
using micrograd::load;
using micrograd::manual_seed;
using micrograd::NoGradGuard;
using micrograd::save;
using micrograd::scalar_t;
using micrograd::Storage;
using micrograd::synchronize;
using micrograd::Tensor;
using micrograd::gpt::Dataset;
using micrograd::gpt::gpt2_124m;
using micrograd::gpt::GPTConfig;
using micrograd::gpt::Model;
using micrograd::gpt::sample_token;
using micrograd::gpt::SampleOrder;
using micrograd::gpt::ShardedDataset;

constexpr size_t kGpt2VocabSize = 50257;

struct TrainConfig {
  std::string data_path = CORPUS_DATA_PATH;
  std::string data_format = "char";
  std::string fineweb_dir = FINEWEB_DATA_DIR;
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
  Device device = Device::CPU;
  bool tf32 = false;
  bool profile = false;
  bool recompute = true;
  micrograd::gpt::ArchOptions arch;
  scalar_t muon_lr = 0.0f;
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
    } else if (arg == "--throughput-interval") {
      config.throughput_interval =
          static_cast<size_t>(std::stoul(next_value()));
    } else if (arg == "--device") {
      config.device = parse_device(next_value());
    } else if (arg == "--tf32") {
      config.tf32 = true;
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
  model_config.recompute_activations = config.recompute;
  model_config.arch = config.arch;
  return model_config;
}

std::shared_ptr<Tensor> token_tensor(const std::vector<size_t> &tokens,
                                     size_t rows, size_t cols, Backend device) {
  std::vector<scalar_t> values(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++) {
    values[i] = static_cast<scalar_t>(tokens[i]);
  }
  auto tensor =
      Tensor::from_values(std::vector<size_t>{rows, cols}, std::move(values));
  tensor->to(device);
  return tensor;
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
  if (!in) {
    throw std::runtime_error("Optimizer state file is truncated");
  }
  return value;
}

using NamedParameters =
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>;

bool is_decayed(const std::string &name) {
  const size_t last_dot = name.rfind('.');
  const size_t leaf_start = last_dot == std::string::npos ? 0 : last_dot + 1;
  return name.substr(leaf_start) == "weight";
}

bool is_muon_matrix(const std::string &name, const Tensor &param) {
  return name.starts_with("h.") && param.shape().size() == 2;
}

class TrainOptimizer {
 public:
  TrainOptimizer(const NamedParameters &named_parameters, scalar_t weight_decay,
                 scalar_t beta1, scalar_t beta2, scalar_t eps,
                 scalar_t base_lr = 0.0f, scalar_t muon_lr = 0.0f)
      : weight_decay_(weight_decay),
        beta1_(beta1),
        beta2_(beta2),
        eps_(eps),
        base_lr_(base_lr),
        muon_lr_(muon_lr) {
    for (const auto &[name, p] : named_parameters) {
      const bool muon = muon_lr > 0.0f && is_muon_matrix(name, *p);
      parameters_.push_back(p);
      decay_.push_back(is_decayed(name));
      muon_.push_back(muon);
      m_.push_back(make_moment(*p));
      v_.push_back(muon ? Tensor::zeros({1}, p->backend()) : make_moment(*p));
    }
  }

  size_t muon_count() const {
    return static_cast<size_t>(std::count(muon_.begin(), muon_.end(), true));
  }

  size_t decayed_count() const {
    return static_cast<size_t>(std::count(decay_.begin(), decay_.end(), true));
  }

  size_t parameter_count() const { return parameters_.size(); }

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
    step_muon(learning_rate);
    if (!parameters_.empty() &&
        parameters_.front()->backend() == Device::CUDA) {
      std::vector<micrograd::cuda::ops::AdamWTensor> tensors;
      tensors.reserve(parameters_.size());
      for (size_t i = 0; i < parameters_.size(); i++) {
        if (muon_[i]) {
          continue;
        }
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
#else
    step_muon(learning_rate);
#endif
    for (size_t i = 0; i < parameters_.size(); i++) {
      if (muon_[i]) {
        continue;
      }
      micrograd::ops::AdamWStep(*parameters_[i], *m_[i], *v_[i], learning_rate,
                                beta1_, beta2_, eps_, weight_decay_, decay_[i],
                                bias_correction1, bias_correction2);
    }
  }

 private:
  void step_muon(scalar_t learning_rate) {
    if (muon_lr_ <= 0.0f) {
      return;
    }
    const scalar_t lr = muon_lr_ * learning_rate / base_lr_;
    for (size_t i = 0; i < parameters_.size(); i++) {
      if (!muon_[i]) {
        continue;
      }
      const auto &shape = parameters_[i]->shape();
      const scalar_t fan_in = static_cast<scalar_t>(shape[0]);
      const scalar_t fan_out = static_cast<scalar_t>(shape[1]);
      const scalar_t scale = std::sqrt(std::max(1.0f, fan_out / fan_in));
      micrograd::ops::MuonStep(*parameters_[i], *m_[i], lr, kMuonMomentum,
                               0.0f, true, kMuonNsSteps, scale);
    }
  }

  static constexpr scalar_t kMuonMomentum = 0.95f;
  static constexpr size_t kMuonNsSteps = 5;

  static std::shared_ptr<Tensor> make_moment(const Tensor &param) {
    auto moment = Tensor::zeros(param.shape());
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
    if (!in) {
      throw std::runtime_error("Optimizer state file is truncated");
    }
    auto loaded = Tensor::from_values(tensor->shape(), std::move(values));
    loaded->to(tensor->backend());
    tensor = loaded;
  }

  std::vector<std::shared_ptr<Tensor>> parameters_;
  std::vector<bool> decay_;
  std::vector<bool> muon_;
  scalar_t weight_decay_;
  scalar_t beta1_;
  scalar_t beta2_;
  scalar_t eps_;
  scalar_t base_lr_;
  scalar_t muon_lr_;
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

std::string data_state_path(const std::string &checkpoint_path) {
  return checkpoint_path + ".data";
}

void save_data_state(const std::string &path, const SampleOrder &order) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for saving: " + path);
  }
  order.write_state(file);
  if (!file) {
    throw std::runtime_error("Failed while writing data state: " + path);
  }
}

void load_data_state(const std::string &path, size_t samples_from_steps,
                     SampleOrder &order) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cout << "no data state at " << path << ", assuming "
              << samples_from_steps
              << " samples consumed from the optimizer step count\n";
    order.seek(samples_from_steps);
    return;
  }
  order.read_state(file);
}

std::string rng_state_path(const std::string &checkpoint_path) {
  return checkpoint_path + ".rng";
}

void save_rng_state(const std::string &path) {
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for saving: " + path);
  }
  file << global_rng();
  if (!file) {
    throw std::runtime_error("Failed while writing RNG state: " + path);
  }
}

void load_rng_state(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cout << "no RNG state at " << path
              << ", dropout will not match the original run\n";
    return;
  }
  std::mt19937_64 rng;
  file >> rng;
  if (!file) {
    throw std::runtime_error("Could not read RNG state: " + path);
  }
  global_rng() = rng;
}

void load_optimizer_state(const std::string &path, TrainOptimizer &optimizer) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for loading: " + path);
  }
  optimizer.read_state(file);
}

std::vector<std::string> find_shards(const std::string &dir,
                                     const std::string &split_name) {
  const std::string marker = "_" + split_name + "_";
  std::vector<std::string> paths;
  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    const std::string name = entry.path().filename().string();
    const bool has_marker = name.find(marker) != std::string::npos;
    const bool is_bin = entry.path().extension() == ".bin";
    if (has_marker && is_bin) {
      paths.push_back(entry.path().string());
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

class TrainingData {
 public:
  explicit TrainingData(const TrainConfig &config) {
    if (config.data_format == "char") {
      chars_ = std::make_unique<Dataset>(config.data_path, config.val_fraction);
      return;
    }
    if (config.data_format != "fineweb") {
      throw std::invalid_argument("Unknown data format: " + config.data_format);
    }
    if (!std::filesystem::is_directory(config.fineweb_dir)) {
      throw std::runtime_error("FineWeb directory not found: " +
                               config.fineweb_dir);
    }
    std::vector<std::string> train_paths =
        find_shards(config.fineweb_dir, "train");
    std::vector<std::string> val_paths = find_shards(config.fineweb_dir, "val");
    if (train_paths.empty() || val_paths.empty()) {
      throw std::runtime_error(
          "FineWeb directory needs train and val shards: " +
          config.fineweb_dir);
    }
    shards_ = std::make_unique<ShardedDataset>(train_paths, val_paths);
  }

  Dataset::Batch sample(size_t batch_size, size_t block_size,
                        Dataset::Split split, std::mt19937_64 &rng) const {
    if (shards_) {
      return shards_->sample(batch_size, block_size, split, rng);
    }
    return chars_->sample(batch_size, block_size, split, rng);
  }

  std::vector<size_t> train_span_tokens() const {
    if (shards_) {
      return shards_->train_span_tokens();
    }
    return chars_->train_span_tokens();
  }

  Dataset::Batch train_batch(size_t batch_size, SampleOrder &order) const {
    if (shards_) {
      return shards_->train_batch(batch_size, order);
    }
    return chars_->train_batch(batch_size, order);
  }

  size_t val_windows(size_t block_size) const {
    if (shards_) {
      return shards_->val_windows(block_size);
    }
    return chars_->val_windows(block_size);
  }

  Dataset::Batch val_batch(size_t batch_size, size_t block_size,
                           size_t batch_index) const {
    if (shards_) {
      return shards_->val_batch(batch_size, block_size, batch_index);
    }
    return chars_->val_batch(batch_size, block_size, batch_index);
  }

  size_t vocab_size() const {
    if (shards_) {
      return kGpt2VocabSize;
    }
    return chars_->vocab_size();
  }

 private:
  std::unique_ptr<Dataset> chars_;
  std::unique_ptr<ShardedDataset> shards_;
};

size_t val_batches(const TrainingData &dataset, const TrainConfig &config) {
  size_t available = dataset.val_windows(config.block_size) / config.batch_size;
  if (available == 0) {
    throw std::runtime_error(
        "Val split holds fewer windows than one batch; lower --batch-size or "
        "--block-size");
  }
  return std::min(config.eval_iters, available);
}

double estimate_loss(Model &model, const TrainingData &dataset,
                     Dataset::Split split, const TrainConfig &config) {
  const NoGradGuard no_grad;
  std::mt19937_64 rng(config.seed);
  const bool is_val = split == Dataset::Split::kVal;
  const size_t batches =
      is_val ? val_batches(dataset, config) : config.eval_iters;
  model.eval();

  const auto rank = static_cast<size_t>(distributed::Rank());
  const auto world = static_cast<size_t>(distributed::WorldSize());
  double total = 0.0;
  for (size_t i = 0; i < batches; i++) {
    Dataset::Batch batch =
        is_val ? dataset.val_batch(config.batch_size, config.block_size, i)
               : dataset.sample(config.batch_size, config.block_size, split,
                                rng);
    if (i % world != rank) {
      continue;
    }
    auto input = token_tensor(batch.inputs, config.batch_size,
                              config.block_size, config.device);
    auto logits = model.forward(input);
    auto flat_logits = logits->reshape(
        {static_cast<int64_t>(config.batch_size * config.block_size),
         static_cast<int64_t>(dataset.vocab_size())});
    auto loss = cross_entropy(flat_logits, batch.targets);
    loss->to(Device::CPU);
    total += static_cast<double>(loss->at({}));
  }

  model.train();
  return distributed::AllReduceSum(total) / static_cast<double>(batches);
}

void run_sample(const TrainConfig &config) {
  if (config.data_format != "char") {
    throw std::invalid_argument(
        "--sample needs --data-format char; BPE decoding is not wired");
  }
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
    auto step_logits = Tensor::from_values(
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

constexpr size_t kThroughputWarmupSteps = 10;
constexpr size_t kProfileWarmupSteps = 3;

double seconds_since(std::chrono::steady_clock::time_point start) {
  auto elapsed = std::chrono::steady_clock::now() - start;
  return std::chrono::duration<double>(elapsed).count();
}

}  // namespace

void skip_samples(SampleOrder &order, size_t count) {
  for (size_t i = 0; i < count; i++) {
    order.next();
  }
}

void sync_gradients(const std::vector<std::shared_ptr<Tensor>> &parameters) {
  if (distributed::WorldSize() == 1) {
    return;
  }
  std::vector<Storage *> grads;
  grads.reserve(parameters.size());
  for (const auto &p : parameters) {
    grads.push_back(&p->grad_storage());
  }
  distributed::AllReduceSum(grads);
}

void broadcast_parameters(
    const std::vector<std::shared_ptr<Tensor>> &parameters) {
  if (distributed::WorldSize() == 1) {
    return;
  }
  for (const auto &p : parameters) {
    distributed::Broadcast(p->data_storage());
  }
}

int main(int argc, char **argv) {
  try {
    TrainConfig config = parse_args(argc, argv);
    distributed::Initialize(distributed::FromEnvironment());
    const auto rank = static_cast<size_t>(distributed::Rank());
    const auto world = static_cast<size_t>(distributed::WorldSize());
    if (!distributed::IsMain()) {
      std::cout.setstate(std::ios::failbit);
    }
    manual_seed(config.seed);

    if (config.sample) {
      run_sample(config);
      return 0;
    }

    if (config.tf32) {
      if (config.device != Device::CUDA) {
        throw std::invalid_argument("--tf32 requires --device cuda");
      }
#ifdef MICROGRAD_CUDA_ENABLED
      micrograd::CudaContext::instance().setAllowTf32(true);
#endif
    }

    TrainingData dataset(config);
    SampleOrder train_order(dataset.train_span_tokens(), config.block_size,
                            config.seed);
    std::cout << "train samples per epoch: " << train_order.samples_per_epoch()
              << "\n";
    std::cout << "val pass: " << val_batches(dataset, config) << " of "
              << dataset.val_windows(config.block_size) / config.batch_size
              << " available batches\n";
    GPTConfig model_config = build_model_config(config);
    Model model(dataset.vocab_size(), model_config);

    std::vector<std::shared_ptr<Tensor>> parameters = model.parameters();
    for (auto &p : parameters) {
      p->to(config.device);
    }
    broadcast_parameters(parameters);
    if (world > 1) {
      std::cout << "data parallel: " << world << " ranks, "
                << distributed::DeviceDescription(config.device)
                << " on rank 0\n";
      manual_seed(config.seed + rank);
    }

    TrainOptimizer optimizer(model.named_parameters(), config.weight_decay,
                             config.beta1, config.beta2, 1e-8f,
                             config.learning_rate, config.muon_lr);
    if (optimizer.muon_count() > 0) {
      std::cout << "muon: " << optimizer.muon_count()
                << " block matrices at lr " << config.muon_lr << "\n";
    }
    const size_t decayed = optimizer.decayed_count();
    const size_t undecayed = optimizer.parameter_count() - decayed;
    std::cout << "weight decay: " << decayed << " tensors decayed, "
              << undecayed << " not\n";

    size_t start_iter = 0;
    if (!config.resume_path.empty()) {
      load(config.resume_path, model);
      load_optimizer_state(optimizer_state_path(config.resume_path), optimizer);
      start_iter = optimizer.step_count();
      load_data_state(
          data_state_path(config.resume_path),
          start_iter * config.batch_size * config.grad_accum_steps * world,
          train_order);
      load_rng_state(rng_state_path(config.resume_path));
      std::cout << "resuming data order at epoch " << train_order.epoch()
                << ", sample " << train_order.position() << "\n";
    }

    std::ofstream log_file;
    if (distributed::IsMain()) {
      log_file.open(config.log_path, config.resume_path.empty()
                                         ? std::ios::out
                                         : std::ios::out | std::ios::app);
    }
    if (distributed::IsMain() && !log_file.is_open()) {
      throw std::runtime_error("Could not open log file: " + config.log_path);
    }
    if (config.resume_path.empty()) {
      log_file << "step,train_loss,val_loss\n";
    }

    const size_t tokens_per_micro_step = config.batch_size * config.block_size;
    const size_t tokens_per_step =
        tokens_per_micro_step * config.grad_accum_steps * world;
    size_t timed_steps = 0;
    size_t profiled_steps = 0;
    double timed_seconds = 0.0;
    double timed_data_seconds = 0.0;

    for (size_t it = start_iter; it < config.max_iters; it++) {
      if (config.eval_interval > 0 && it % config.eval_interval == 0) {
        double train_loss =
            estimate_loss(model, dataset, Dataset::Split::kTrain, config);
        double val_loss =
            estimate_loss(model, dataset, Dataset::Split::kVal, config);
        std::cout << "step " << it << ": train loss " << train_loss
                  << ", val loss " << val_loss << "\n";
        log_file << it << "," << train_loss << "," << val_loss << "\n";
        log_file.flush();
      }

      if (distributed::IsMain() && config.checkpoint_interval > 0 && it > 0 &&
          it % config.checkpoint_interval == 0) {
        save(config.checkpoint_path, model);
        save_optimizer_state(optimizer_state_path(config.checkpoint_path),
                             optimizer);
        save_data_state(data_state_path(config.checkpoint_path), train_order);
        save_rng_state(rng_state_path(config.checkpoint_path));
      }

      if (config.profile && it == start_iter + kProfileWarmupSteps) {
        micrograd::profiler::Enable();
      }

      const auto step_start = std::chrono::steady_clock::now();
      model.train();
      {
        micrograd::profiler::Scope scope("zero_grad", "step", config.device);
        optimizer.zero_grad();
      }
      double step_data_seconds = 0.0;
      for (size_t micro_step = 0; micro_step < config.grad_accum_steps;
           micro_step++) {
        const auto data_start = std::chrono::steady_clock::now();
        skip_samples(train_order, rank * config.batch_size);
        Dataset::Batch batch =
            dataset.train_batch(config.batch_size, train_order);
        skip_samples(train_order, (world - 1 - rank) * config.batch_size);
        std::shared_ptr<Tensor> input;
        {
          micrograd::profiler::Scope scope("batch_to_device", "step",
                                           config.device);
          input = token_tensor(batch.inputs, config.batch_size,
                               config.block_size, config.device);
        }
        step_data_seconds += seconds_since(data_start);
        auto logits = model.forward(input);
        auto flat_logits = logits->reshape(
            {static_cast<int64_t>(config.batch_size * config.block_size),
             static_cast<int64_t>(dataset.vocab_size())});
        auto loss = cross_entropy(flat_logits, batch.targets);
        auto scaled_loss =
            loss->div(static_cast<scalar_t>(config.grad_accum_steps * world));
        scaled_loss->backward();
      }
      {
        micrograd::profiler::Scope scope("grad_sync", "step", config.device);
        sync_gradients(parameters);
      }
      {
        micrograd::profiler::Scope scope("grad_clip", "step", config.device);
        micrograd::ops::ClipGradNorm(parameters, config.grad_clip);
      }
      {
        micrograd::profiler::Scope scope("optimizer", "step", config.device);
        optimizer.step(LrAt(it, config));
      }
      if (micrograd::profiler::Enabled()) {
        micrograd::profiler::Flush();
        profiled_steps++;
      }

      if (config.throughput_interval == 0) {
        continue;
      }
      synchronize(config.device);
      if (it - start_iter < kThroughputWarmupSteps) {
        continue;
      }
      timed_seconds += seconds_since(step_start);
      timed_data_seconds += step_data_seconds;
      timed_steps++;
      if (timed_steps == config.throughput_interval) {
        double seconds_per_step =
            timed_seconds / static_cast<double>(timed_steps);
        double tokens_per_second =
            static_cast<double>(tokens_per_step) / seconds_per_step;
        double data_ms_per_step =
            timed_data_seconds * 1000.0 / static_cast<double>(timed_steps);
        std::cout << "step " << it << ": " << tokens_per_second << " tok/s, "
                  << seconds_per_step * 1000.0 << " ms/step, data "
                  << data_ms_per_step << " ms/step ("
                  << 100.0 * timed_data_seconds / timed_seconds << "%)\n";
        timed_steps = 0;
        timed_seconds = 0.0;
        timed_data_seconds = 0.0;
      }
    }

    if (config.profile) {
      std::cout << micrograd::profiler::Report(profiled_steps);
      micrograd::profiler::Disable();
    }
#ifdef MICROGRAD_CUDA_ENABLED
    if (config.device == Device::CUDA) {
      std::cout << "gpu memory: "
                << micrograd::CudaContext::instance().memorySummary() << "\n";
    }
#endif

    double final_val_loss =
        estimate_loss(model, dataset, Dataset::Split::kVal, config);
    std::cout << "final val loss " << final_val_loss << "\n";

    if (distributed::IsMain()) {
      save(config.checkpoint_path, model);
      save_optimizer_state(optimizer_state_path(config.checkpoint_path),
                           optimizer);
      save_data_state(data_state_path(config.checkpoint_path), train_order);
      save_rng_state(rng_state_path(config.checkpoint_path));
      std::cout << "Saved trained model to " << config.checkpoint_path << "\n";
    }
    distributed::Barrier();
    distributed::Shutdown();
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
