# micrograd
A small autograd engine

## API

### Tensor
```C++
// Create a 4x4 tensor
Tensor t({4,4});

// Set and Get values
t.at({2,2}) = 23.00;
t.at({2, 2}) // 23.00

// Accessors
t.size() // 16
t.shape() // {4,4}
```