#include "conv1d.h"
#include "conv2d.h"

void conv1d(float* input, float* kernel, float* output,
            int N, int K, int S, int P, int D)
{
    // Conv1dのパラメータをConv2dに適応
    const int channels = 1;
    const int num_kernels = 1;
    const int height = 1;
    const int width = N;
    const int kernel_h = 1;
    const int kernel_w = K;
    const int pad_h = 0;
    const int pad_w = P;
    const int stride_h = 1;
    const int stride_w = S;
    const int dilation_h = 1;
    const int dilation_w = D;

    // Conv2dを直接呼び出し、outputに結果を格納
    conv2d(input, channels, height, width,
           kernel, num_kernels, kernel_h, kernel_w,
           pad_h, pad_w, stride_h, stride_w,
           dilation_h, dilation_w, output);
}

