#include <iostream>

#include "micrograd/Tensor.h"

int main() {
  Tensor t({4, 4});
  t.at({2, 2}) = 42.0;
  std::cout << t.at({2, 2}) << std::endl;
  return 0;
}
