# micrograd

A small automatic differentiation engine.

```c++
#include "micrograd/Tensor.h"
#include <iostream>

int main() {
  auto prediction = std::make_shared<Tensor>(
    std::vector<size_t>{2},
    std::vector<double>{1, 2});

  auto target = std::make_shared<Tensor>(
    std::vector<size_t>{2},
    std::vector<double>{3, 3});

  auto loss = prediction->sub(target)->pow(2.0)->sum();

  loss->backward();

  std::cout << loss->data()[0] << "\n"; // 5

  for (auto& v : prediction->grad()){
    std::cout << v << " "; // -4 -2
  }
}
```

## Build & Run

```bash
cmake -S . -B build
cmake --build build
./build/main
```

`main` trains a small network on MNIST.

## Tests

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```
