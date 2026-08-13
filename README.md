# micrograd

A small automatic differentiation engine.

## Build & Run

```bash
cmake -S . -B build
cmake --build build
./build/main
```

## Tests
Tests compare values computed from Pytorch to verify correctness.

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```


## Example
```c++
#include "micrograd/Tensor.h"
#include <iostream>

int main() {
  auto a = std::make_shared<Tensor>(
    std::vector<size_t>{2, 2},
    std::vector<double>{1, 2, 3, 4});

  auto b = std::make_shared<Tensor>(
    std::vector<size_t>{2, 2},
    std::vector<double>{5, 6, 7, 8});

  auto c = a->matmul(b);
  auto loss = c->sum();

  loss->backward();

  for (auto& v : c->data()){
    std::cout << v << " "; // 19 22 43 50
  }

  for (auto& v : a->grad()){
    std::cout << v << " "; // 11 15 11 15
  }
}
```
