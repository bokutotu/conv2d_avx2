#pragma once

void conv2d(const float* data_im, const int channels,
            const int height, const int width,
            const float* data_kernels, const int num_kernels,
            const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w,
            const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w,
            float* data_out);
