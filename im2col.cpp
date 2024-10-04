#include <immintrin.h>
#include <cstring>

void im2col(const float* data_im, const int channels,
            const int height, const int width,
            const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w,
            const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w,
            float* data_col)
{
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    const int channel_size = height * width;
    const int kernel_size = kernel_h * kernel_w;

    #pragma omp parallel for collapse(2)
    for (int c = 0; c < channels; ++c)
    {
        for (int kh = 0; kh < kernel_h; ++kh)
        {
            for (int kw = 0; kw < kernel_w; ++kw)
            {
                int im_row_base = -pad_h + kh * dilation_h;
                for (int oh = 0; oh < output_h; ++oh)
                {
                    int im_row = im_row_base + oh * stride_h;
                    int im_col_base = -pad_w + kw * dilation_w;

                    if (im_row >= 0 && im_row < height)
                    {
                        int col_index = ((c * kernel_size + kh * kernel_w + kw) * output_h + oh) * output_w;
                        int im_index = c * channel_size + im_row * width;

                        int ow = 0;
                        for (; ow <= output_w - 8; ow += 8)
                        {
                            __m256i im_cols = _mm256_setr_epi32(
                                im_col_base + (ow + 0) * stride_w,
                                im_col_base + (ow + 1) * stride_w,
                                im_col_base + (ow + 2) * stride_w,
                                im_col_base + (ow + 3) * stride_w,
                                im_col_base + (ow + 4) * stride_w,
                                im_col_base + (ow + 5) * stride_w,
                                im_col_base + (ow + 6) * stride_w,
                                im_col_base + (ow + 7) * stride_w);

                            __m256i mask = _mm256_and_si256(
                                _mm256_cmpgt_epi32(im_cols, _mm256_set1_epi32(-1)),
                                _mm256_cmpgt_epi32(im_cols, _mm256_set1_epi32(width)));

                            int valid_mask = _mm256_movemask_epi8(mask);
                            if (valid_mask == -1)
                            {
                                // All columns are valid
                                __m256 data = _mm256_i32gather_ps(
                                    data_im + im_index, im_cols, 4);
                                _mm256_storeu_ps(data_col + col_index + ow, data);
                            }
                            else
                            {
                                // Handle invalid columns
                                float tmp[8];
                                for (int i = 0; i < 8; ++i)
                                {
                                    int im_col = im_col_base + (ow + i) * stride_w;
                                    if (im_col >= 0 && im_col < width)
                                    {
                                        tmp[i] = data_im[im_index + im_col];
                                    }
                                    else
                                    {
                                        tmp[i] = 0.0f;
                                    }
                                }
                                _mm256_storeu_ps(data_col + col_index + ow, _mm256_loadu_ps(tmp));
                            }
                        }
                        // Handle remaining columns
                        for (; ow < output_w; ++ow)
                        {
                            int im_col = im_col_base + ow * stride_w;
                            if (im_col >= 0 && im_col < width)
                            {
                                data_col[col_index + ow] = data_im[im_index + im_col];
                            }
                            else
                            {
                                data_col[col_index + ow] = 0.0f;
                            }
                        }
                    }
                    else
                    {
                        // Set entire row to zero if im_row is invalid
                        int col_index = ((c * kernel_size + kh * kernel_w + kw) * output_h + oh) * output_w;
                        std::memset(data_col + col_index, 0, sizeof(float) * output_w);
                    }
                }
            }
        }
    }
}

