#include "conv2d.h"
#include "conv1d.h"

#include <cstdlib>
#include <cstring>
#include <random>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <iomanip>

struct Conv2d {
private:
  float* kernel_ptr;
  int channels;
  int num_kernels;
  int kernel_h;
  int kernel_w;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;

  void random_init_kernel() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < num_kernels * channels * kernel_h * kernel_w; i++) {
      kernel_ptr[i] = dis(gen);
    }
  }

public:
  Conv2d(const float* data_kernels, const int channels, const int num_kernels,
         const int kernel_h, const int kernel_w,
         const int pad_h, const int pad_w,
         const int stride_h, const int stride_w,
         const int dilation_h, const int dilation_w)
    : channels(channels), num_kernels(num_kernels), kernel_h(kernel_h), kernel_w(kernel_w),
      pad_h(pad_h), pad_w(pad_w), stride_h(stride_h), stride_w(stride_w),
      dilation_h(dilation_h), dilation_w(dilation_w)
  {
    kernel_ptr = static_cast<float*>(malloc(num_kernels * channels * kernel_h * kernel_w * sizeof(float)));
    if (data_kernels) {
      std::copy(data_kernels, data_kernels + num_kernels * channels * kernel_h * kernel_w, kernel_ptr);
    } else {
      random_init_kernel();
    }
  }

  void SetKernel(const float* data_kernels) {
    std::copy(data_kernels, data_kernels + num_kernels * channels * kernel_h * kernel_w, kernel_ptr);
  }

  void forward(const float* data_im, const int height, const int width, float* data_out) {
    conv2d(data_im, channels, height, width, kernel_ptr, num_kernels, kernel_h, kernel_w,
           pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, data_out);
  }

  ~Conv2d() {
    free(kernel_ptr);
  }
};

// Function to generate random float data
std::vector<float> generate_random_data(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  std::vector<float> data(size);
  for (auto& val : data) {
      val = static_cast<float>(dis(gen));
  }
  return data;
}

void print_array(const std::vector<float>& arr, const std::string& name) {
  std::cout << name << ": [";
  for (size_t i = 0; i < arr.size(); ++i) {
    std::cout << std::fixed << std::setprecision(2) << arr[i];
    if (i < arr.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
}

void run_test(int N, int K, int S, int P, int D) {
  // 入力とカーネルの初期化
  std::vector<float> input(N);
  std::vector<float> kernel(K);
  for (int i = 0; i < N; ++i) input[i] = (i % 10) + 1; // 1から10の繰り返し
  for (int i = 0; i < K; ++i) kernel[i] = (i % 3 + 1) * 0.1f; // 0.1, 0.2, 0.3の繰り返し

  // 出力サイズの計算
  int output_size = (N + 2 * P - D * (K - 1) - 1) / S + 1;
  std::vector<float> output(output_size);

  // conv1d関数の呼び出し
  conv1d(input.data(), kernel.data(), output.data(), N, K, S, P, D);

  // 結果の表示
  std::cout << "Conv1D Test Results:" << std::endl;
  std::cout << "Parameters: N=" << N << ", K=" << K << ", S=" << S 
            << ", P=" << P << ", D=" << D << std::endl;
  print_array(input, "Input");
  print_array(kernel, "Kernel");
  print_array(output, "Output");
  std::cout << std::endl;
}

void benchmark_conv1d(int N, int K, int S, int P, int D, int num_runs) {
  // 入力とカーネルの初期化
  std::vector<float> input(N);
  std::vector<float> kernel(K);
  
  // ランダムな値で初期化
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  for (int i = 0; i < N; ++i) input[i] = dis(gen);
  for (int i = 0; i < K; ++i) kernel[i] = dis(gen);

  // 出力サイズの計算
  int output_size = (N + 2 * P - D * (K - 1) - 1) / S + 1;
  std::vector<float> output(output_size);

  // ベンチマーク実行
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_runs; ++i) {
      conv1d(input.data(), kernel.data(), output.data(), N, K, S, P, D);
  }
  auto end = std::chrono::high_resolution_clock::now();

  // 実行時間の計算
  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / num_runs;

  // 結果の表示
  std::cout << "C++ Conv1D Benchmark Results:" << std::endl;
  std::cout << "Parameters: N=" << N << ", K=" << K << ", S=" << S 
            << ", P=" << P << ", D=" << D << std::endl;
  std::cout << "Number of runs: " << num_runs << std::endl;
  std::cout << "Average execution time: " << avg_time << " ms" << std::endl;
  std::cout << std::endl;
}

// Benchmark function
// void run_benchmark(int batch_size, int in_channels, int out_channels, 
//                    int input_height, int input_width, int kernel_size, 
//                    int num_iterations) {
void run_benchmark() {
  // Define benchmark parameters
  const int batch_size = 32;
  const int in_channels = 64;
  const int out_channels = 128;
  const int input_height = 224;
  const int input_width = 224;
  const int kernel_size = 3;
  const int num_iterations = 10;
  const int pad = 1;
  const int stride = 1;
  const int dilation = 1;

  // Generate random input data
  std::vector<float> input = generate_random_data(batch_size * in_channels * input_height * input_width);

  // Generate random filter data
  std::vector<float> filter = generate_random_data(out_channels * in_channels * kernel_size * kernel_size);

  // Create Conv2d object
  Conv2d conv2d(filter.data(), in_channels, out_channels, kernel_size, kernel_size,
                pad, pad, stride, stride, dilation, dilation);

  // Calculate output dimensions
  int output_height = input_height;  // Because of padding
  int output_width = input_width;    // Because of padding

  // Allocate memory for output
  std::vector<float> output(batch_size * out_channels * output_height * output_width);

  // Perform benchmark
  std::cout << "Starting benchmark..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_iterations; ++i) {
    conv2d.forward(input.data(), input_height, input_width, output.data());
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  // Print benchmark results
  std::cout << "Benchmark results:" << std::endl;
  std::cout << "Input size: " << batch_size << "x" << in_channels << "x" << input_height << "x" << input_width << std::endl;
  std::cout << "Filter size: " << out_channels << "x" << in_channels << "x" << kernel_size << "x" << kernel_size << std::endl;
  std::cout << "Output size: " << batch_size << "x" << out_channels << "x" << output_height << "x" << output_width << std::endl;
  std::cout << "Number of iterations: " << num_iterations << std::endl;
  std::cout << "Total time: " << diff.count() << " seconds" << std::endl;
  std::cout << "Average time per iteration: " << diff.count() / num_iterations << " seconds" << std::endl;

  // Optionally, print a small portion of the output to verify correctness
  std::cout << "\nFirst few output values:" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << std::setprecision(6) << std::fixed << output[i] << " ";
  }
  std::cout << std::endl;
}

int main() {
  run_test(60, 6, 3, 3, 1);

  benchmark_conv1d(10000, 101, 1, 50, 1, 100);

  const int batch_size = 1;
  const int in_channels = 2;
  const int out_channels = 3;
  const int input_height = 4;
  const int input_width = 4;
  const int kernel_height = 3;
  const int kernel_width = 3;
  const int pad_h = 1;
  const int pad_w = 1;
  const int stride_h = 1;
  const int stride_w = 1;
  const int dilation_h = 1;
  const int dilation_w = 1;

  float input[32] = {
    0.5432947,
    -0.39515755,
    0.20552567,
    -0.45032975,
    -0.5730771,
    -0.5553584,
    0.59432304,
    1.5419426,
    1.8197253,
    -0.5515287,
    -1.325326,
    0.18855357,
    -0.069072686,
    -0.49492535,
    -1.4959149,
    -0.19383712,
    -0.4731198,
    0.33555076,
    1.5091219,
    2.0819554,
    1.7067116,
    2.3803675,
    -1.1256016,
    -0.3169981,
    -0.14067143,
    0.8057536,
    0.3276143,
    -0.7607072,
    -1.599082,
    0.018486667,
    -0.7504268,
    0.18540798,
  };

  float output[48] = {
    0.3671525,
    -0.17387724,
    -0.53952014,
    -0.41356063,
    0.13519445,
    -0.6369239,
    -0.5777169,
    -0.07820636,
    -0.6019154,
    -0.85000455,
    -0.227178,
    0.38553098,
    0.53258127,
    0.4952766,
    0.16334829,
    0.5179188,
    -1.1829954,
    -0.15092221,
    0.15374796,
    0.5376092,
    -0.35269666,
    -0.10102463,
    -0.628401,
    -0.40036133,
    -0.5694187,
    -0.1765114,
    -0.05552435,
    -0.3107502,
    -0.6736164,
    -0.44401115,
    -0.1804393,
    0.056986123,
    0.5652461,
    0.8913239,
    0.30458608,
    -0.7666081,
    0.15480474,
    0.14275207,
    0.42336845,
    0.12534592,
    0.5706087,
    0.40240055,
    -0.16282544,
    -0.032061294,
    0.47645676,
    -0.09869753,
    -0.34638345,
    -0.02880986,
  };

  float filter[54] = {
    -0.0017646605,
    0.12644097,
    -0.1939936,
    -0.1734625,
    -0.090781756,
    0.063205294,
    -0.0046700113,
    0.18688585,
    -0.020917172,
    0.06236978,
    -0.071232304,
    -0.046330906,
    -0.2251778,
    -0.15610139,
    -0.09716192,
    0.008731253,
    0.0931814,
    0.14142673,
    -0.15979224,
    -0.10263957,
    0.0856111,
    0.19572432,
    -0.048507567,
    0.17637877,
    -0.03799128,
    0.024940623,
    0.21342279,
    -0.218654,
    -0.14838351,
    -0.05967162,
    -0.09187673,
    0.20364694,
    -0.1527774,
    -0.1085015,
    -0.16467114,
    -0.22074954,
    -0.13758895,
    0.2026092,
    0.105174676,
    0.11423842,
    0.01239595,
    -0.12084066,
    0.039877214,
    -0.22007395,
    -0.1703105,
    -0.121511586,
    0.1487135,
    0.13819724,
    -0.104532786,
    -0.0085047,
    0.1507459,
    0.23431942,
    0.093546025,
    0.03184169,
  };

  Conv2d conv2d(filter, in_channels, out_channels, kernel_height, kernel_width,
                  pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);

  int output_height = input_height;  // Because of padding
  int output_width = input_width;    // Because of padding
  std::vector<float> output_pred(batch_size * out_channels * output_height * output_width);

  conv2d.forward(input, input_height, input_width, output_pred.data());

  for (int i = 0; i < output_pred.size(); i++) {
    if (std::abs(output_pred[i] - output[i]) > 1e-6) {
      std::cerr << "Mismatch at index " << i << ": " << output_pred[i] << " != " << output[i] << std::endl;
      return 1;
    }
  }
  std::cout << "Test passed!" << std::endl;

  run_benchmark();

  return 0;
}
