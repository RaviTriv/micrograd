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
t3 = t->add(2.0);
// Subtract
auto t3 = t->sub(t2);
t4 = t->sub(2.0);
// Multiply
auto t3 = t->mul(t2);
t3 = t->mul(2.0);
// Divide
auto t3 = t->div(t2);
t3 = t->mul(2.0)
// Exponent
auto t3 = t->pow(3.0);

// Matrix Multiplication
auto t3 = t->matmul(t2);

// Set and Get values
t.at({2,2}) = 23.00;
t.at({2, 2}) // 23.00

// Accessors
t.size() // 16
t.shape() // {4,4}
```