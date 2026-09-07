#pragma once

namespace micrograd {

bool GradEnabled();

class NoGradGuard {
 public:
  NoGradGuard();
  ~NoGradGuard();
  NoGradGuard(const NoGradGuard &) = delete;
  NoGradGuard &operator=(const NoGradGuard &) = delete;
  NoGradGuard(NoGradGuard &&) = delete;
  NoGradGuard &operator=(NoGradGuard &&) = delete;

 private:
  bool previous_;
};

}  // namespace micrograd
