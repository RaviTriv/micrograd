#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "micrograd/Device.h"

namespace micrograd::profiler {

struct Entry {
  std::string name;
  std::string phase;
  size_t calls = 0;
  size_t nested_calls = 0;
  double total_ms = 0.0;
};

struct Summary {
  double wall_ms = 0.0;
  double busy_ms = 0.0;
  std::vector<Entry> entries;
};

void Enable();
void Disable();
bool Enabled();
void Flush();
Summary Collect();
std::string Report(size_t steps);

class Scope {
 public:
  Scope(const char *name, const char *phase, Device device);
  ~Scope();

  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;
  Scope(Scope &&) = delete;
  Scope &operator=(Scope &&) = delete;

 private:
  const char *name_;
  const char *phase_;
  [[maybe_unused]] Device device_;
  bool active_ = false;
  bool top_level_ = false;
  double cpu_start_ms_ = 0.0;
  [[maybe_unused]] size_t slot_ = 0;
};

}  // namespace micrograd::profiler
