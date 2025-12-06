# micrograd
A small autograd engine

## API

### Tensor
```C++
// Create a 2x2 tensor
auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 2},
                                  std::vector<double>{1.0, 2.0, 3.0, 4.0});
auto t2 = std::make_shared<Tensor>(std::vector<size_t>{2, 2},
                                   std::vector<double>{5.0, 6.0, 7.0, 8.0});

// Add
auto t3 = t->add(t2);
// Subtract
auto t4 = t->sub(t2);
// Multiply
auto t5 = t->mul(t2);
// Divide
auto t6 = t->div(t2);

// Set and Get values
t.at({2,2}) = 23.00;
t.at({2, 2}) // 23.00

// Accessors
t.size() // 16
t.shape() // {4,4}
```