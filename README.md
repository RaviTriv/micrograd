# micrograd
A small automatic differentiation engine.

## Computational Graphs
Mathematical expressions can be represented as directed acyclic graphs with nodes as variables or functions and edges as data flow.

![Computational Graph](./images/graph.png)

## Automatic Differentiation (Autodiff)
Autodiff is a method to automatically compute derivatives by breaking functions into smaller ops and applying the chain rule.

## Build & Run
```bash
mkdir build && cd build
cmake ..
make
./main
```

## MNIST Demo
See [here](https://ravtrive.com/micrograd-demo) for demo of micrograd trained MNIST Model used for inference of drawn digits.
![Digit 5](./images/mnist_demo_5.png)
![Digit 7](./images/mnist_demo_7.png)
![Digit 9](./images/mnist_demo_9.png)

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

## Accelerators
Accelerators can be leveraged to parallelize ops making execution a lot faster!
Ops are dispatched to compute shaders and launch a gpu thread per element to execute in parallel.

### Execution Pipeline

### Parallel Reduction

### Supported Accelerators
- Metal


## Tests
Tests compare values computed from Pytorch to verify correctness.
```bash
cd build && cmake .. -DBUILD_TESTS=ON && make micrograd_tests && ctest --output-on-failure
```
