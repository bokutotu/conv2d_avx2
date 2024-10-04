#include <cblas.h>
#include <cstring>

#include "im2col.h"

void conv2d(const float* data_im, const int channels,
            const int height, const int width,
            const float* data_kernels, const int num_kernels,
            const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w,
            const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w,
            float* data_out)
{
    // compute output dimensions
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    // size of the im2col matrix
    const int k_size = channels * kernel_h * kernel_w;
    const int n_size = output_h * output_w;

    // allocate memory for im2col output
    float* data_col = new float[k_size * n_size];

    // perform im2col transformation
    im2col(data_im, channels, height, width,
           kernel_h, kernel_w, pad_h, pad_w,
           stride_h, stride_w, dilation_h, dilation_w,
           data_col);

    // reshape data_kernels into a matrix of size (num_kernels, k_size)
    float* weights_col = new float[num_kernels * k_size];

    #pragma omp parallel for
    for (int n = 0; n < num_kernels; ++n)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int kh = 0; kh < kernel_h; ++kh)
            {
                for (int kw = 0; kw < kernel_w; ++kw)
                {
                    int k_index = ((n * channels + c) * kernel_h + kh) * kernel_w + kw;
                    int col_index = n * k_size + ((c * kernel_h + kh) * kernel_w + kw);
                    weights_col[col_index] = data_kernels[k_index];
                }
            }
        }
    }

    // perform matrix multiplication using blas
    // data_out = weights_col (num_kernels x k_size) * data_col (k_size x n_size)
    // resulting in data_out (num_kernels x n_size)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                num_kernels, n_size, k_size,
                1.0f, weights_col, k_size,
                data_col, n_size,
                0.0f, data_out, n_size);

    // cleanup allocated memory
    delete[] data_col;
    delete[] weights_col;
}

