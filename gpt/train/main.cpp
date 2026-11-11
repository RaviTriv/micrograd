#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
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
#include "gpt/ModelFile.h"
#include "gpt/TrainConfig.h"
#include "gpt/TrainOptimizer.h"
#include "gpt/TrainState.h"
#include "micrograd/Autograd.h"
#include "micrograd/Distributed.h"
#include "micrograd/NN.h"
#include "micrograd/Profiler.h"
#include "micrograd/Random.h"
#include "micrograd/Scalar.h"
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
using micrograd::load;
using micrograd::manual_seed;
using micrograd::NoGradGuard;
using micrograd::scalar_t;
using micrograd::synchronize;
using micrograd::Tensor;
using micrograd::gpt::data_state_path;
using micrograd::gpt::check_train_state;
using micrograd::gpt::checkpoint_set_path;
using micrograd::gpt::read_latest_checkpoint;
using micrograd::gpt::remove_checkpoint_set;
using micrograd::gpt::resolve_resume_path;
using micrograd::gpt::write_latest_checkpoint;
using micrograd::gpt::Dataset;
using micrograd::gpt::GPTConfig;
using micrograd::gpt::load_data_state;
using micrograd::gpt::load_optimizer_state;
using micrograd::gpt::load_rng_state;
using micrograd::gpt::LrAt;
using micrograd::gpt::Model;
using micrograd::gpt::optimizer_state_path;
using micrograd::gpt::parse_args;
using micrograd::gpt::read_model_config;
using micrograd::gpt::require_same_architecture;
using micrograd::gpt::rng_state_path;
using micrograd::gpt::sample_token;
using micrograd::gpt::SampleOrder;
using micrograd::gpt::save_data_state;
using micrograd::gpt::save_model;
using micrograd::gpt::save_optimizer_state;
using micrograd::gpt::save_rng_state;
using micrograd::gpt::ShardedDataset;
using micrograd::gpt::TrainConfig;
using micrograd::gpt::TrainOptimizer;

constexpr size_t kGpt2VocabSize = 50257;

constexpr size_t kVocabPadMultiple = 128;

GPTConfig build_model_config(const TrainConfig &config, size_t vocab_size) {
  GPTConfig model_config;
  model_config.depth = config.n_layer;
  model_config.n_layer = config.n_layer;
  model_config.n_embd = config.n_embd;
  model_config.n_head = config.n_head;
  model_config.n_kv_head = config.n_head;
  model_config.block_size = config.block_size;
  model_config.vocab_size = vocab_size;
  model_config.dropout = config.dropout;
  model_config.matrix_lr = 0.0f;
  model_config.embedding_lr = 0.0f;
  model_config.unembedding_lr = 0.0f;
  model_config.recompute_activations = config.recompute;
  model_config.arch = config.arch;
  if (config.pad_vocab) {
    model_config.padded_vocab_size = (vocab_size + kVocabPadMultiple - 1) /
                                     kVocabPadMultiple * kVocabPadMultiple;
  }
  if (config.bf16) {
    model_config.activation_dtype = micrograd::DType::kBFloat16;
  }
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
    auto loss = cross_entropy(logits, batch.targets, dataset.vocab_size());
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
  GPTConfig model_config = build_model_config(config, dataset.vocab_size());
  Model model(dataset.vocab_size(), model_config);

  std::vector<std::shared_ptr<Tensor>> parameters = model.parameters();
  for (auto &p : parameters) {
    p->to(config.device);
  }
  require_same_architecture(read_model_config(config.checkpoint_path),
                            model_config);
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
    auto logits = model.forward(input)->cast(micrograd::DType::kFloat32);
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

void save_checkpoint(const TrainConfig &config, Model &model,
                     const GPTConfig &model_config,
                     const micrograd::gpt::TrainOptimizer &optimizer,
                     const SampleOrder &train_order,
                     std::optional<size_t> &kept_step) {
  const size_t step = optimizer.step_count();
  const std::string set = checkpoint_set_path(config.checkpoint_path, step);
  save_model(set, model, model_config);
  save_optimizer_state(optimizer_state_path(set), optimizer);
  save_data_state(data_state_path(set), train_order, step);
  save_rng_state(rng_state_path(set), step);
  write_latest_checkpoint(config.checkpoint_path, step);
  if (kept_step && *kept_step != step) {
    remove_checkpoint_set(
        checkpoint_set_path(config.checkpoint_path, *kept_step));
  }
  kept_step = step;
}

void skip_samples(SampleOrder &order, size_t count) {
  for (size_t i = 0; i < count; i++) {
    order.next();
  }
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
    TrainConfig defaults;
    defaults.data_path = CORPUS_DATA_PATH;
    defaults.fineweb_dir = FINEWEB_DATA_DIR;
    TrainConfig config = parse_args(argc, argv, defaults);
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

    if (config.bf16_matmul) {
      if (config.device != Device::CUDA) {
        throw std::invalid_argument("--bf16-matmul requires --device cuda");
      }
#ifdef MICROGRAD_CUDA_ENABLED
      micrograd::CudaContext::instance().setAllowBf16Matmul(true);
#endif
    }

    if (config.bf16 && config.device != Device::CUDA) {
      throw std::invalid_argument("--bf16 requires --device cuda");
    }

    const std::string flags = micrograd::gpt::format_flags(config);
    std::cout << "flags: " << flags << "\n";

    TrainingData dataset(config);
    SampleOrder train_order(dataset.train_span_tokens(), config.block_size,
                            config.seed);
    std::cout << "train samples per epoch: " << train_order.samples_per_epoch()
              << "\n";
    std::cout << "val pass: " << val_batches(dataset, config) << " of "
              << dataset.val_windows(config.block_size) / config.batch_size
              << " available batches\n";
    GPTConfig model_config = build_model_config(config, dataset.vocab_size());
    Model model(dataset.vocab_size(), model_config);
    if (model_config.padded_vocab_size > dataset.vocab_size()) {
      std::cout << "vocab: " << dataset.vocab_size() << " padded to "
                << model_config.padded_vocab_size << "\n";
    }
    if (config.bf16) {
      std::cout << "activations: bf16, fp32 master weights\n";
    }

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
    distributed::GradientReducer reducer(
        parameters, config.grad_bucket_mb << 20, config.bf16_grad_reduce);
    if (reducer.overlaps()) {
      std::cout << "gradient all-reduce: " << reducer.bucket_count()
                << " buckets of about " << config.grad_bucket_mb << " MB"
                << (config.bf16_grad_reduce ? " in bf16" : "")
                << ", overlapped with backward\n";
    }

    TrainOptimizer optimizer(model.named_parameters(), config.weight_decay,
                             config.beta1, config.beta2, 1e-8f,
                             config.learning_rate, config.muon_lr,
                             config.bf16 && !config.muon_fp32);
    if (optimizer.muon_count() > 0) {
      std::cout << "muon: " << optimizer.muon_count()
                << " block matrices at lr " << config.muon_lr
                << (config.bf16 && !config.muon_fp32 ? ", bf16 Newton-Schulz"
                                                     : "")
                << "\n";
    }
    const size_t decayed = optimizer.decayed_count();
    const size_t undecayed = optimizer.parameter_count() - decayed;
    std::cout << "weight decay: " << decayed << " tensors decayed, "
              << undecayed << " not\n";

    size_t start_iter = 0;
    std::optional<size_t> kept_step =
        read_latest_checkpoint(config.checkpoint_path);
    if (!config.resume_path.empty()) {
      const std::string resume_path = resolve_resume_path(config.resume_path);
      require_same_architecture(read_model_config(resume_path), model_config);
      const size_t saved_step = check_train_state(resume_path);
      load(resume_path, model);
      load_optimizer_state(optimizer_state_path(resume_path), optimizer,
                           saved_step);
      start_iter = optimizer.step_count();
      load_data_state(
          data_state_path(resume_path),
          start_iter * config.batch_size * config.grad_accum_steps * world,
          train_order, saved_step);
      load_rng_state(rng_state_path(resume_path), saved_step);
      std::cout << "resuming from " << resume_path << " at step " << saved_step
                << "\n";
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
    log_file << "# flags: " << flags << "\n";
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
        save_checkpoint(config, model, model_config, optimizer, train_order,
                        kept_step);
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
        reducer.prepare(micro_step + 1 == config.grad_accum_steps);
        auto logits = model.forward(input);
        auto loss = cross_entropy(logits, batch.targets, dataset.vocab_size());
        auto scaled_loss =
            loss->div(static_cast<scalar_t>(config.grad_accum_steps * world));
        scaled_loss->backward();
        loss->to(Device::CPU);
        micrograd::gpt::check_finite_loss(loss->at({}), it);
      }
      {
        micrograd::profiler::Scope scope("grad_sync", "step", config.device);
        reducer.finish();
      }
      const bool fused_clip = optimizer.fuses_grad_clip();
      if (!fused_clip) {
        micrograd::profiler::Scope scope("grad_clip", "step", config.device);
        micrograd::ops::ClipGradNorm(parameters, config.grad_clip);
      }
      {
        micrograd::profiler::Scope scope("optimizer", "step", config.device);
        if (fused_clip) {
          optimizer.step(LrAt(it, config), config.grad_clip, parameters);
        } else {
          optimizer.step(LrAt(it, config));
        }
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
      save_checkpoint(config, model, model_config, optimizer, train_order,
                      kept_step);
      save_model(config.checkpoint_path, model, model_config);
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
