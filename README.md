# SIMD Conv2D Project

This project demonstrates an efficient implementation of the 2D convolution operation using SIMD (Single Instruction, Multiple Data) instructions and OpenMP for parallelization. It serves as both an example and a practice exercise for optimizing computational kernels in deep learning applications.

## Features

- Implements 2D convolution using im2col and GEMM (General Matrix Multiplication) approach
- Utilizes SIMD instructions (AVX2) for vectorized operations
- Employs OpenMP for multi-threading
- Includes a benchmark utility to measure performance

## Requirements

- C++11 compatible compiler
- CMake (version 3.10 or higher)
- OpenMP support
- BLAS library (e.g., OpenBLAS, Intel MKL)
- AVX2 support in the CPU

## Building the Project

1. Clone the repository:
   ```
   git clone git@github.com:bokutotu/conv2d_avx2.git
   cd conv2d_avx2
   ```

2. Create a build directory and navigate to it:
   ```
   mkdir build
   cd build
   ```

3. Run CMake and build the project:
   ```
   cmake ..
   make
   ```

## Running the Program

After building, you can run the executable:

```
./conv2d_executable
```

This will run a small test case to verify the correctness of the implementation and then perform a benchmark.

## Implementation Details

The project consists of three main components:

1. `im2col.cpp`: Implements the im2col operation, which rearranges image blocks to columns for efficient GEMM operation.
2. `conv2d.cpp`: Implements the main convolution operation using im2col and BLAS GEMM.
3. `main.cpp`: Contains the `Conv2d` class, test cases, and benchmarking code.

The implementation uses SIMD instructions (AVX2) in the im2col operation for improved performance. OpenMP is used for parallelization in both im2col and the kernel reshaping step in conv2d.

## Benchmarking

The project includes a benchmarking function that measures the performance of the convolution operation. It runs the convolution multiple times and reports the average execution time.

## Contributing

Contributions to improve the implementation or extend the project are welcome. Please feel free to submit pull requests or open issues for any bugs or enhancements.

## License

This project is open-source and available under the MIT License.

## Acknowledgments

This project was created as a learning exercise and demonstration of SIMD and parallel computing techniques in the context of deep learning operations.
