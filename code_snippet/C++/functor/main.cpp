#include <iostream>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <functional>

// 学习函子相关的使用
// 函子表现为一个定义了operator()的类--》创建一个看起来像函数的对象
// 可以将add_x当一个函数来使用
struct add_x{
  add_x(int val) : x(val) {}
  int operator()(int y) const {
    return x + y;
  }
private:
  int x;
};

// 普通的成员函数
struct add_x2 {
  add_x2(int val) : x(val) {}
  int add(int y) const {
    return x + y;
  }

  static int add_x4(int y) {
    return 20 + y;
  }

private:
  int x;
};

// 普通的函数
int add_x3( int y) {
  return y + 20;
}

int main() {
  //test1 : functor
  add_x add30(30);
  int fn = add30(6);
  assert(fn == 36);

  std::vector<int> in(10, 0), out(10);
  std::transform(in.begin(), in.end(), out.begin(), add_x(1));
  for(int i = 0; i < in.size(); ++i) {
    assert(out[i] == in[i] + 1);
  }
  std::transform(in.begin(), in.end(), out.begin(), add30);
  for(int i = 0; i < in.size(); ++i) {
    assert(out[i] == in[i] + 30);
  }

  // test2 : member fun
  add_x2 add20(20);
 // int (add_x2::*fn2) (int) const = &add_x2::add;
  typedef int (add_x2::*FN2) (int) const;
  FN2 fn2 = &add_x2::add;                  //终于写对了
  int res = (add20.*fn2)(6);
  assert(res == 26);

  std::vector<int> in2(10, 0), out2(10);

  // test3 : fun
  typedef int (*FN3)(int);
  FN3 fn3 = add_x3;
  res = fn3(6);
  assert(res == 26);

  std::vector<int> in3(10, 0), out3(10);
  std::transform(in3.begin(), in3.end(), out3.begin(), fn3);
  for(int i = 0; i < in3.size(); ++i) {
    assert(out3[i] == in3[i] + 20);
  }
  std::transform(in3.begin(), in3.end(), out3.begin(), add_x3);
  for(int i = 0; i < in3.size(); ++i) {
    assert(out3[i] == in3[i] + 20);
  }

  // test4 : static member fun
  typedef int (*FN4) (int);
  FN4 fn4 = &add_x2::add_x4;
  res = fn4(6);
  assert(res == 26);

  std::vector<int> in4(10, 0), out4(10);
  std::transform(in4.begin(), in4.end(), out4.begin(), fn4);
  for(int i = 0; i < in4.size(); ++i) {
    assert(out4[i] == in4[i] + 20);
  }

  // test5 : lambda
  std::vector<int> in5(10, 0), out5(10);
  std::transform(in5.begin(), in5.end(), out5.begin(), [] (int y) { return 20 + y;});
  for(int i = 0; i < in5.size(); ++i) {
    assert(out5[i] == in5[i] + 20);
  }

  // test6 : boost::function bind member fun
  add_x2 add10(10);
  boost::function<int (int)> fb1(boost::bind(&add_x2::add, &add10, _1));
  // C++11 functional
  std::function<int (int)> fun1(std::bind(&add_x2::add, &add10, std::placeholders::_1));
  res = fb1(6);
  assert(res == 16);

  std::vector<int> in6(10, 0), out6(10);
  std::transform(in6.begin(), in6.end(), out6.begin(), fb1);
  for(int i = 0; i < in6.size(); ++i) {
    assert(out6[i] == in6[i] + 10);
  }

  // test7 : boost::function bind static member fun
  boost::function<int (int)> fb2(boost::bind(&add_x2::add_x4, _1));
  // C++11 functional，stl库也已经实现了类似的功能
  std::function<int (int)> fun2(std::bind(&add_x2::add_x4, std::placeholders::_1));
  res = fb2(6);
  assert(res == 26);

  std::vector<int> in7(10, 0), out7(10);
  std::transform(in7.begin(), in7.end(), out7.begin(), fb2);
  for(int i = 0; i < in7.size(); ++i) {
    assert(out7[i] == in7[i] + 20);
  }

  //boost::bind可以包装函数指针也可以包装函子，但是包装函子感觉没什么意义
  add_x add40(40);
  boost::function<int (int)> fb3(add40);


  std::cout << "Hello, World!" << std::endl;
  return 0;
}
