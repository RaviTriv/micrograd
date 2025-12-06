#include <iostream>

#include "micrograd/Tensor.h"

int main() {
  auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 2},
                                    std::vector<double>{1.0, 2.0, 3.0, 4.0});
  auto t2 = std::make_shared<Tensor>(std::vector<size_t>{2, 2},
                                     std::vector<double>{5.0, 6.0, 7.0, 8.0});

  auto t3 = t->add(t2);
  auto t4 = t->sub(t2);
  auto t5 = t->mul(t2);
  auto t6 = t->div(t2);

  std::cout << "ADD: " << t3->at({1, 1}) << std::endl;
  std::cout << "SUB: " << t4->at({1, 1}) << std::endl;
  std::cout << "MUL: " << t5->at({1, 1}) << std::endl;
  std::cout << "DIV: " << t6->at({1, 1}) << std::endl;

  t3 = t->add(2.0);
  t4 = t->sub(2.0);
  t5 = t->mul(2.0);
  t6 = t->div(2.0);
  std::cout << "ADD SCALAR: " << t3->at({1, 1}) << std::endl;
  std::cout << "SUB SCALAR: " << t4->at({1, 1}) << std::endl;
  std::cout << "MUL SCALAR: " << t5->at({1, 1}) << std::endl;
  std::cout << "DIV SCALAR: " << t6->at({1, 1}) << std::endl;

  t2 = t->pow(3.0);
  std::cout << "POW SCALAR: " << t2->at({1, 1}) << std::endl;

  return 0;
}
