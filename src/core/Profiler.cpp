#include "micrograd/Profiler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef MICROGRAD_CUDA_ENABLED
#include "micrograd/backends/cuda/CudaCheck.h"
#include "micrograd/backends/cuda/CudaContext.h"
#endif

namespace micrograd::profiler {
namespace {

double NowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Totals {
  size_t calls = 0;
  size_t nested_calls = 0;
  double total_ms = 0.0;
};

#ifdef MICROGRAD_CUDA_ENABLED
struct PendingGpu {
  const char *name;
  const char *phase;
  bool top_level;
  cudaEvent_t start;
  cudaEvent_t end;
};
#endif

struct State {
  bool enabled = false;
  int depth = 0;
  std::thread::id owner;
  double wall_start_ms = 0.0;
  double busy_ms = 0.0;
  std::map<std::pair<std::string, std::string>, Totals> totals;
#ifdef MICROGRAD_CUDA_ENABLED
  std::vector<PendingGpu> pending;
  std::vector<cudaEvent_t> free_events;
#endif
};

State &GetState() {
  static State state;
  return state;
}

void CheckOwnerThread([[maybe_unused]] const State &state) {
#ifndef NDEBUG
  if (state.owner != std::this_thread::get_id()) {
    throw std::logic_error(
        "profiler is single-threaded, use it only from the thread that "
        "called Enable");
  }
#endif
}

void AddTiming(State &state, const char *name, const char *phase,
               bool top_level, double ms) {
  Totals &totals = state.totals[{name, phase}];
  totals.calls++;
  totals.total_ms += ms;
  if (top_level) {
    state.busy_ms += ms;
  } else {
    totals.nested_calls++;
  }
}

void SynchronizeDevices() {
#ifdef MICROGRAD_CUDA_ENABLED
  CudaContext &ctx = CudaContext::instance();
  if (ctx.isAvailable()) {
    ctx.synchronize();
  }
#endif
}

#ifdef MICROGRAD_CUDA_ENABLED
cudaEvent_t AcquireEvent(State &state) {
  if (!state.free_events.empty()) {
    cudaEvent_t event = state.free_events.back();
    state.free_events.pop_back();
    return event;
  }
  cudaEvent_t event = nullptr;
  MICROGRAD_CUDA_CHECK(cudaEventCreate(&event));
  return event;
}
#endif

}  // namespace

void Enable() {
  State &state = GetState();
  state.owner = std::this_thread::get_id();
  Flush();
  SynchronizeDevices();
  state.totals.clear();
  state.busy_ms = 0.0;
  state.enabled = true;
  state.wall_start_ms = NowMs();
}

void Disable() {
  CheckOwnerThread(GetState());
  Flush();
  GetState().enabled = false;
}

bool Enabled() { return GetState().enabled; }

void Flush() {
#ifdef MICROGRAD_CUDA_ENABLED
  State &state = GetState();
  if (state.depth > 0 || state.pending.empty()) {
    return;
  }
  MICROGRAD_CUDA_CHECK(cudaEventSynchronize(state.pending.back().end));
  for (const PendingGpu &pending : state.pending) {
    float ms = 0.0f;
    MICROGRAD_CUDA_CHECK(cudaEventElapsedTime(&ms, pending.start, pending.end));
    AddTiming(state, pending.name, pending.phase, pending.top_level,
              static_cast<double>(ms));
    state.free_events.push_back(pending.start);
    state.free_events.push_back(pending.end);
  }
  state.pending.clear();
#endif
}

Summary Collect() {
  CheckOwnerThread(GetState());
  Flush();
  SynchronizeDevices();
  State &state = GetState();

  Summary summary;
  summary.wall_ms = NowMs() - state.wall_start_ms;
  summary.busy_ms = state.busy_ms;
  for (const auto &[key, totals] : state.totals) {
    summary.entries.push_back({.name = key.first,
                               .phase = key.second,
                               .calls = totals.calls,
                               .nested_calls = totals.nested_calls,
                               .total_ms = totals.total_ms});
  }
  std::ranges::sort(summary.entries, [](const Entry &a, const Entry &b) {
    return a.total_ms > b.total_ms;
  });
  return summary;
}

std::string Report(size_t steps) {
  const Summary summary = Collect();
  const double per_step = steps > 0 ? static_cast<double>(steps) : 1.0;
  const double wall = summary.wall_ms > 0.0 ? summary.wall_ms : 1.0;

  std::string out;
  char line[256];
  std::snprintf(line, sizeof(line),
                "profile over %zu steps: wall %.2f ms/step, timed %.2f "
                "ms/step (%.1f%%), outside timed scopes %.2f ms/step\n",
                steps, summary.wall_ms / per_step, summary.busy_ms / per_step,
                100.0 * summary.busy_ms / wall,
                (summary.wall_ms - summary.busy_ms) / per_step);
  out += line;
  std::snprintf(line, sizeof(line), "%-26s %-5s %11s %10s %10s %7s\n", "op",
                "phase", "calls/step", "ms/step", "avg us", "% wall");
  out += line;
  bool any_nested = false;
  for (const Entry &entry : summary.entries) {
    const bool nested = entry.nested_calls > 0;
    any_nested = any_nested || nested;
    std::snprintf(
        line, sizeof(line), "%-26s %-5s %11.1f %10.3f %10.1f %6.1f%s\n",
        entry.name.c_str(), entry.phase.c_str(),
        static_cast<double>(entry.calls) / per_step, entry.total_ms / per_step,
        1000.0 * entry.total_ms / static_cast<double>(entry.calls),
        100.0 * entry.total_ms / wall, nested ? "*" : "");
    out += line;
  }
  if (any_nested) {
    out +=
        "* some calls ran inside another timed row and are also counted "
        "there\n";
  }
  return out;
}

Scope::Scope(const char *name, const char *phase, Device device)
    : name_(name), phase_(phase), device_(device) {
  State &state = GetState();
  if (!state.enabled) {
    return;
  }
  CheckOwnerThread(state);
  active_ = true;
  top_level_ = state.depth == 0;
  state.depth++;
#ifdef MICROGRAD_CUDA_ENABLED
  if (device_ == Device::CUDA) {
    slot_ = state.pending.size();
    state.pending.push_back({.name = name_,
                             .phase = phase_,
                             .top_level = top_level_,
                             .start = AcquireEvent(state),
                             .end = AcquireEvent(state)});
    MICROGRAD_CUDA_CHECK(cudaEventRecord(state.pending[slot_].start,
                                         CudaContext::instance().stream()));
    return;
  }
#endif
  cpu_start_ms_ = NowMs();
}

Scope::~Scope() {
  if (!active_) {
    return;
  }
  State &state = GetState();
  state.depth--;
#ifdef MICROGRAD_CUDA_ENABLED
  if (device_ == Device::CUDA) {
    cudaEventRecord(state.pending[slot_].end, CudaContext::instance().stream());
    return;
  }
#endif
  AddTiming(state, name_, phase_, top_level_, NowMs() - cpu_start_ms_);
}

}  // namespace micrograd::profiler
