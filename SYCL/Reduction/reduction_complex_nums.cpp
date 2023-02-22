// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <complex>
#include <numeric>

#include <sycl/sycl.hpp>

using namespace sycl;

// Currently, Identityless reduction for complex numbers is
// only valid for plus operator.
// TODO: Extend this test case once we support known_identity for std::complex
// and more operators (apart from plus).
template <typename T>
void test_identityless_reduction_for_complex_nums(queue &q) {
  // Allocate and initialize buffer on the host with all 1's.
  buffer<std::complex<T>> valuesBuf{1024};
  {
    host_accessor a{valuesBuf};
    std::fill(a.begin(), a.end(), std::complex<T>(1));
  }

  // Buffer to hold the reduction results.
  std::complex<T> sumResult = 0;
  buffer<std::complex<T>> sumBuf{&sumResult, 1};

  q.submit([&](handler &cgh) {
    auto inputVals = valuesBuf.template get_access<access_mode::read>(cgh);
    auto sumReduction = reduction(sumBuf, cgh, plus<std::complex<T>>());

    cgh.parallel_for(range<1>{1024}, sumReduction,
                     [=](id<1> idx, auto &sum) { sum += inputVals[idx]; });
  });

  assert(sumBuf.get_host_access()[0] == static_cast<T>(1024));
}

int main() {
  queue q;

  test_identityless_reduction_for_complex_nums<float>(q);
  test_identityless_reduction_for_complex_nums<double>(q);

  return 0;
}
