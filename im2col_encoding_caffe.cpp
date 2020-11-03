
#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
// #include <cblas.h>

#ifdef __cplusplus
extern "C"
{
#endif
   #include <cblas.h>
#ifdef __cplusplus
}
#endif

// https://www.programmersought.com/article/16963235/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;
 
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}
 
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const Dtype* A, const Dtype* B, const float beta,
    Dtype* C) {
  int lda = (TransA == CblasNoTrans) ? K : M; // The size of the first dimention of matrix A; if you are passing a matrix A[m][n], the value should be m.
  int ldb = (TransB == CblasNoTrans) ? N : K; // The size of the first dimention of matrix A; if you are passing a matrix A[m][n], the value should be m.


  // std::cout << "lda: " << lda << " " << ldb << " " << TransA << " " << TransB << " " << CblasNoTrans << " " << K << " " << M << " " << N << std::endl;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
	if (alpha == 0) {
		memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
		return;
	}
	for (int i = 0; i < N; ++i) {
		Y[i] = alpha;
	}
}
 
template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	Dtype* data_col) {
	const int output_h = (height + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	
	const int channel_size = height * width;
	for (int channel = channels; channel--; data_im += channel_size) {
		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_h + kernel_row * dilation_h;
				for (int output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						for (int output_cols = output_w; output_cols; output_cols--) {
							*(data_col++) = 0;
						}
					}
					else {
						int input_col = -pad_w + kernel_col * dilation_w;
						for (int output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								*(data_col++) = data_im[input_row * width + input_col];
								
							}
							else {
								*(data_col++) = 0;
							}
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}
	}

}
 
 
template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	Dtype* data_im) {
	caffe_set(height * width * channels, Dtype(0), data_im);
	const int output_h = (height + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	const int channel_size = height * width;
	for (int channel = channels; channel--; data_im += channel_size) {
		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_h + kernel_row * dilation_h;
				for (int output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						data_col += output_w;
					}
					else {
						int input_col = -pad_w + kernel_col * dilation_w;
						for (int output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								data_im[input_row * width + input_col] += *data_col;
							}
							data_col++;
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}
	}
}
 
// If you want to run a 6x6 matrix, please uncomment the comment below and comment out the 5X5 section.

/*
float dataim[] = {
	1,2,3,4,5,6,
	5,6,7,8,9,10,
	6,5,4,3,2,1,
	10,9,8,7,6,5,
	4,3,2,1,5,6,
	3,2,1,6,5,4,
};
*/

// data all ones 6x6
float dataim[] = {
	1,1,1,1,1,1,
	1,1,1,1,1,1,
	1,1,1,1,1,1,
	1,1,1,1,1,1,
	1,1,1,1,1,1,
	1,1,1,1,1,1,
};

/**/

/*
 float dataim[] = {
 	1,2,3,4,5,
 	6,7,8,9,10,
 	5,4,3,2,1,
 	10,9,8,7,6,
 	4,3,2,1,5,
 };
*/ 

float datacol[1000];
float outim[50];
 
int main()
{
	//im2col_cpu(dataim, 1, 6, 6, 3, 3, 0, 0, 1, 1, 1, 1, datacol);
	//col2im_cpu(datacol, 1, 6, 6, 3, 3, 0, 0, 1, 1, 1, 1, outim);
	
	 // int Ih = 5;
	 // int Iw = 5;

	int Ic = 1;
	int Ih = 6;
	int Iw = 6;

	int Kh = 3;
	int Kw = 3;

	int Sh = 1;
	int Sw = 1;
	int dilation_h = 1;
	int dilation_w = 1;	

	int pad_h = 0;
	int pad_w = 0;

	int output_h = (Ih + 2 * pad_h -
		(dilation_h * (Kh - 1) + 1)) / Sh + 1;
	int output_w = (Iw + 2 * pad_w -
		(dilation_w * (Kw - 1) + 1)) / Sw + 1;

//	int output_h = (1 + Ih - Kh + 2 * pad_h)/Sh; // removed + 1
//        int output_w = (1 + Iw - Kw + 2 * pad_w)/Sw;


	// int im2col_height = output_h*output_w;
	// int im2col_width  = Kh*Kw*Ic;
	
	int im2col_height  = Kh*Kw*Ic;
	int im2col_width =  output_h*output_w;

	im2col_cpu(dataim, 1, Ih, Iw, Kh, Kw, 0, 0, Sh, Sw, dilation_h, dilation_w, datacol);

	// im2col_cpu(dataim, 1, 6, 6, 3, 3, 0, 0, 1, 1, 1, 1, datacol);
	// col2im_cpu(datacol, 1, 5, 5, 3, 3, 0, 0, 1, 1, 1, 1, outim);

	float* output = new float[output_h*output_w];
	float* filter = new float[Kh*Kw];
	
	// filter setting
	for(int i = 0; i < Kh*Kw; ++i)
	{
	  filter[i] = 1;
	}	
	
	cout << "Original Image: " << endl;
	for(int i=0; i < Ih; i++)
	{
		for(int j=0; j< Iw; j++)
		{
			cout << dataim[j + i*Iw] << " " ;
		}

		cout << "\n";
	}
	
	cout << "1D Data:" << endl;
	for (int i = 0; i < Ih*Iw; ++i)
	{
	 cout << dataim[i] << " " ;
	}
	cout << "\n";

	/*
	for(int i=0; i < Ih*Iw; i++)
	{
		
		cout << outim[i] << " " ;
		cout << "\n";
	}
	*/
	
	cout << "Im2col: " << im2col_height << ", " << im2col_width << endl;
	for(int i = 0; i < im2col_height; ++i)
	{
	 for(int j = 0; j < im2col_width; ++j)
	 {
	 	cout << datacol[j + i*im2col_width] << " " ;
	 }
		cout << "\n";
	}

/*	for(int i=0; i < im2col_height*im2col_width; ++i)
	{
	 cout << datacol[i] << endl;
	}

*/
	int conv_out_channels_ = 1;
	int conv_out_spatial_dim_ = output_h*output_w;
	int weight_offset_ = Kh*Kw;
	
	// caffe: group_ = this->layer_param_.convolution_param().group();	
	// caffe: weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
	// caffe: col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
	// caffe: output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
	// CHECK_EQ(channels_ % group_, 0);
       //   CHECK_EQ(num_output_ % group_, 0)
       // << "Number of output should be multiples of group.";
  
	int col_offset_    = im2col_height*im2col_width;
	int output_offset_ = conv_out_spatial_dim_;
	
	for (int g = 0; g < Ic; ++g) {
   		 caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        	Ic, conv_out_spatial_dim_, Kh*Kw,
        	(float)1., filter + weight_offset_ * g, datacol + col_offset_ * g,
        	(float)0., output + output_offset_ * g);
		
  	}
	
	cout << "Conv Output: " << output_h << ", " << output_w << endl;
	for(int i = 0; i < conv_out_spatial_dim_; ++i)
	{
		cout << output[i] << endl;
	}

	delete [] output;
	delete [] filter;
	return 0;
}
 
 // If you want to run a 5x5 matrix, please uncomment the comment below and comment out the above paragraph
/* 
int dataim[] = {
	1,2,3,4,5,
	6,7,8,9,10,
	5,4,3,2,1,
	10,9,8,7,6,
	4,3,2,1,5,
};
int datacol[1000];
int outim[50];
int main()
{
	im2col_cpu(dataim, 1, 5, 5, 3, 3, 0, 0, 1, 1, 1, 1, datacol);
	col2im_cpu(datacol, 1, 5, 5, 3, 3, 0, 0, 1, 1, 1, 1, outim);
	return 0;
*/
