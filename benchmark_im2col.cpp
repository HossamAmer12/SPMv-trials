


#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <boost/timer/timer.hpp>
#include <time.h>
#include <fstream>


// https://www.programmersought.com/article/16963235/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Set percision
#include <iomanip>



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


using namespace Eigen;
using namespace std;
using namespace boost::timer;


void print_caffe_out(float* output, int conv_out_spatial_dim_)
{
	for(int i = 0; i < conv_out_spatial_dim_; ++i)
	{
		cout << output[i] << endl;
	}

}
 
double generate_org_featureMap(MatrixXf& org_fm, int Ih, int Iw, double density)
{
    for (int i = 0; i < ceil(density * Ih * Iw); ++i)
    {
        int r = rand() % Ih;
        int c = rand() % Iw;
        if (org_fm(r, c) == 0)
        {
            org_fm(r, c) = 1;
        }
        else
        {
            bool found = false;
            for (int u = 0; u < Ih; ++u)
            {
                for (int v = 0; v < Iw; ++v)
                {
                    if (org_fm(u, v) == 0)
                    {
                        org_fm(u, v) = 1;
                        found = true;
                        break;
                    }
                }
                if (found)
                    break;
            }
        }
    }

    // Calculate the actual dennsity
    double density_cal = 0;
    for (unsigned i = 0; i < Ih; ++i)
    {
        for (unsigned j = 0; j < Iw; ++j)
        {
            if (org_fm(i, j) != 0) {
                density_cal += 1;
            }
        }
    }

    density_cal = density_cal / (Ih * Iw);
    return density_cal;

}

void Im2col_Encoding(MatrixXf &im2col_mat, MatrixXf& org_fm, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, int Ic)
{
    int start_row_int = 0;
    int start_col_int = 0;
   
    int patch_matrix_index = 0;
    // For each patch:
    for (int patch = 0; patch < Oh * Ow; patch = patch + Sw)
    {
        int im2col_mat_col_index = 0;

        // Fetch this piece (all rows, all cols in this current submatrix)
        for (int row_int = start_row_int; row_int < start_row_int + Kh; ++row_int)
        {
            for (int col_int = start_col_int; col_int < start_col_int + Kw; ++col_int)
            {
                im2col_mat(patch_matrix_index, im2col_mat_col_index) = org_fm(row_int, col_int);
                im2col_mat_col_index++;
            } // end inner loop
        } // end outer loop
        patch_matrix_index++;
        // increment the start row
        start_row_int = start_row_int + Sw;

        if (start_row_int + Kh > Ih)
        {
            start_row_int = 0;
            start_col_int = start_col_int + Sw;
        }
        if (start_col_int + Kw > Iw)
        {
            break;
        }
    } // end outer outer loop
}

void reset_Im2col_Encoding(MatrixXf &im2col_mat)
{
    int start_row_int = 0;
    int start_col_int = 0;
   
    for(int i = 0; i < im2col_mat.rows() ; ++i)
    {
      for(int j = 0; j < im2col_mat.cols(); ++j)
      {
        im2col_mat(i, j) = 0;
      }
    }
}


void reset_Im2col_Encoding_Caffe(float* dataim, int Ih, int Iw)
{
    for(int i = 0; i < Ih*Iw ; ++i)
    {
        *(dataim + i) = 0;
    }
}

void bench_Dense(const MatrixXf &m, const MatrixXf &in, MatrixXf &o) {
    o.noalias() = m*in;    
}


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
 

bool isErrorIm2col(const MatrixXf &im2col_mat, float* datacol, int im2col_height, int im2col_width)
{
	int total_size_caffe = im2col_height*im2col_width;
	int total_size_mine  = im2col_mat.rows()*im2col_mat.cols();
   	if(total_size_mine != total_size_caffe)
   	{
   		cout << "Dims Mismatch " << im2col_mat.rows() << ", " << im2col_mat.cols() << " | " << im2col_height << ", " << im2col_width <<endl;
   		return true;
   	}	

 //    std::cout << "\n===My Out Map (" << im2col_height << "x" << im2col_width <<  "):  \n" << std::endl;
    
 //    int count_print = 0;
 //    for(int i = 0; i < im2col_height ; ++i)
 //    {
 //      for(int j = 0; j < im2col_width; ++j)
 //      {
 //      	cout << im2col_mat(count_print++, 0) << ", ";
 //      }
 //      cout << "\n";
 //  }

 //    std::cout << "\n===Caffe Out Map (" << im2col_height << "x" << im2col_width <<  "):  \n" << std::endl;
	// for(int i = 0; i < im2col_height ; ++i)
	// {
	//   for(int j = 0; j < im2col_width; ++j)
	//   {
	//   	cout << datacol[i + j*im2col_width] << ", ";
	//   }
	//   cout << "\n";
	// }

 //   cout << "-----\n" << endl;
 

   	int count = 0;
    for(int i = 0; i < im2col_height ; ++i)
	{
	  for(int j = 0; j < im2col_width; ++j)
	  {
	  	if(datacol[i + j*im2col_width] != im2col_mat(count++, 0))
	  	{
	  		cout << "Values Mismatch " << im2col_mat(count-1, 0) << " => " << datacol[i + j*im2col_width] << endl;
        	// cout << " At my index: " << i << ", " << j << ", 1D: " << j+i*im2col_width << endl;
        	// cout << "Oh: " << im2col_mat.cols() << " Ow: " << im2col_mat.rows() << " output_h: " << im2col_height << " output_w: " << im2col_width << endl;  
        	return true;
	  	}
	  }
	}

    return false;
}



int main()
{
	// // bench iterations
	int bench_iterations = 100000;

    // Big test
   // std::vector<int> I_list = {50, 8, 17};
   // std::vector<int> Kh_list = {3, 1, 3, 7, 1};
   // std::vector<int> Kw_list = {3, 3, 1, 1, 7};

  std::vector<int> I_list = {8};
  std::vector<int> Kh_list = {3};
  std::vector<int> Kw_list = {3};

   for(int KK = 0; KK < Kh_list.size(); ++KK)
   {
   	for(int II = 0; II < I_list.size(); ++II)
   	{
   		int Ih = I_list[II];
        int Iw = I_list[II];

        // density:
        // float density = 0.1;
        float density = 0.05;
   //         float density = 0.3;
   //      float density = 1;
        for(; density < 1.05; density+=0.05)
        {
        	// timer for im2col, csr
		    float t_im2col = 0;
		    float t_im2col_caffe    = 0;
		    
		    // timer for creations:
		    float t_im2col_creation = 0;
		    float t_im2col_creation_caffe = 0;

		    // Conv parameters:
		    int padding = 0;
		    int stride  = 1;
		    int Sh, Sw;
		    Sh = Sw = stride;
		    int num_filters = 1; // 6

   		    int dilation_h = 1;
		    int dilation_w = 1;	
 

		    int Kh = Kh_list[KK];
		    int Kw = Kw_list[KK];

		    // usleess case skip it
		    if(Ih == 8) if((Kh == 1 && Kw == 7) || Kh == 7) continue;

			int Ic = 1; // put it as articial for now
			int In = 1;

			int K = 1; // number of filters

			int Oh = (1 + Ih - Kh + 2 * padding)/stride; // removed + 1
			int Ow = (1 + Iw - Kw + 2 * padding)/stride;

			int pad_h = 0;
			int pad_w = 0;

			int output_h = (Ih + 2 * pad_h -
			(dilation_h * (Kh - 1) + 1)) / Sh + 1;
			int output_w = (Iw + 2 * pad_w -
			(dilation_w * (Kw - 1) + 1)) / Sw + 1;

			int im2col_height  = Kh*Kw*Ic;
			int im2col_width =  output_h*output_w;


			// Create your original input feature map:
			MatrixXf org_fm = MatrixXf::Zero(Ih, Iw);
			double density_cal = generate_org_featureMap(org_fm, Ih, Iw, density);

			// Prepare the output for im2col, sparseMat
			MatrixXf d_o1 = MatrixXf::Zero(Oh, Ow);

			// Decalare Eigen Vector
			VectorXf filter_vectorized  = VectorXf::Ones(Kh*Kw);

			// Prepare the output for CPO
			vector<vector<float> > O( Oh , vector<float> (Ow, 0));

			// Create the Kernel
			vector<int> Kernel(Kh*Kw, 1);


			// Create the float data for im2col
			float* dataim = new float[Ih*Iw];

			for(int i = 0; i < Ih; ++i)
			{
				for(int j = 0; j < Iw; ++j)
				{
					dataim[j + i*Iw] = org_fm(i, j);	
				}
			}

			// Define the output and filter for caffe [Note: Remember this data should be deleted below - look at code below!!!]
			float* output = new float[output_h*output_w];
			float* filter = new float[Kh*Kw];

			// Create the data col
			float* datacol = new float[im2col_height*im2col_width];

			// filter setting
			for(int i = 0; i < Kh*Kw; ++i)
			{
			  filter[i] = 1;
			}


			// ====================================== Im2Col My Version ========================


			// Create the intermediate representation for im2col:
			MatrixXf im2col_mat    = MatrixXf::Zero(Oh*Ow, Kh*Kw*Ic);
			{
				for(int k = 0; k < bench_iterations; ++k)
				{

				  clock_t t_im2col_creation_c;
				  t_im2col_creation_c = clock();
				  Im2col_Encoding(im2col_mat, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, Ic);
				  double elapsed  = 1000*((double)(clock()-t_im2col_creation_c))/CLOCKS_PER_SEC; // time in milliseconds
				  if(k > 0)
				    t_im2col_creation += elapsed/(Ih*Iw*1.0); // normalized timing

				  // Reset im2col encoing
				  if(k != bench_iterations-1)
				  {
				    reset_Im2col_Encoding(im2col_mat);
				  }
				} // end for loop
			} // end scope

			// Perform 50 times dense matrix dense vector multiplication: d_o1 = d_m * d_b
			{

				for(int k=0; k<bench_iterations; k++){

				    clock_t t;
				    t = clock();
				    bench_Dense(im2col_mat, filter_vectorized, d_o1);

				    double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds

				     if (k > 0)
				      t_im2col += elapsed/(Ih*Iw*1.0); // normalized timing
				} // end for loop
			} // end scope 

			// include creation time:
			t_im2col +=  t_im2col_creation;

			// ====================================== Im2Col Caffe Version ========================
			{
				for(int k = 0; k < bench_iterations; ++k)
				{

				  clock_t t_im2col_creation_c;
				  t_im2col_creation_c = clock();
				  im2col_cpu(dataim, Ic, Ih, Iw, Kh, Kw, pad_h, pad_w, Sh, Sw, dilation_h, dilation_w, datacol);
				  double elapsed  = 1000*((double)(clock()-t_im2col_creation_c))/CLOCKS_PER_SEC; // time in milliseconds
				  if(k > 0)
				    t_im2col_creation_caffe += elapsed/(Ih*Iw*1.0); // normalized timing

				  // Reset im2col encoing
				  if(k != bench_iterations-1)
				  {
				    reset_Im2col_Encoding_Caffe(datacol, Ih, Iw);  
				  }
				} // end for loop
			} // end scope

			int conv_out_channels_ = 1;
			int conv_out_spatial_dim_ = output_h*output_w;
			int weight_offset_ = Kh*Kw;

			int col_offset_    = im2col_height*im2col_width;
			int output_offset_ = conv_out_spatial_dim_;

			{
				for(int k = 0; k < bench_iterations; ++k)
				{
					clock_t t;
					t = clock();

					for (int g = 0; g < Ic; ++g) {
						caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
						Ic, conv_out_spatial_dim_, Kh*Kw,
						(float)1., filter + weight_offset_ * g, datacol + col_offset_ * g,
						(float)0., output + output_offset_ * g);

					}

					// print_caffe_out(output, conv_out_spatial_dim_);
					// print_caffe_out(datacol, im2col_width*im2col_height);
					// print_caffe_out(dataim, Ih*Iw);

					double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
					if (k > 0)
						t_im2col_caffe += elapsed/(Ih*Iw*1.0); // normalized timing

				} // end for loop

				// include creation time:
				t_im2col_caffe +=  t_im2col_creation_caffe;

			} // end scope

			//-----

			ofstream myfile;
			myfile.open ("csr_log.txt", ios::out | ios::app);


			bool is_error = isErrorIm2col(d_o1, output, output_h, output_w);

			ofstream myfile_encoding;
			myfile_encoding.open ("encoding_log.txt", ios::out | ios::app);
			myfile << std::setprecision(3) << density  << "\t" << density_cal << "\t" << Kh << "\t" << Kw << "\t" << Ih << "\t" << Iw 
			<< "\t" << t_im2col 
			<< "\t" << t_im2col_caffe 
			// << "\t" << 100.0*(t_im2col_caffe-t_im2col)/t_im2col 
			<< "\n";

			myfile_encoding << std::setprecision(3) << density  << "\t" << density_cal << "\t" << Kh << "\t" << Kw << "\t" << Ih << "\t" << Iw 
			<< "\t" << t_im2col_creation 
			<< "\t" << t_im2col_creation_caffe
			// << "\t" << 100.0*(t_im2col_creation_caffe-t_im2col_creation)/t_im2col_creation 
			<< "\n";

			myfile.close();
			myfile_encoding.close();

			std::cout << std::setprecision(3) << density  << "\t" << density_cal << "\t" << Kh << "\t" << Kw << "\t" << Ih << "\t" << Iw 
			<< "\t" << t_im2col 
			<< "\t" << t_im2col_caffe 
			<< "\t"  <<std::boolalpha << !is_error
			// << "\t" << 100.0*(t_im2col_caffe-t_im2col)/t_im2col 
			<< "\n";



			// Delete the data created
			delete [] dataim;
			delete [] output;
			delete [] filter;
			delete [] datacol;

		} // end denisty loop
		} // end I loop

		ofstream myfile;
	    myfile.open ("csr_log.txt", ios::out | ios::app);
	    myfile << "\n";
	    myfile.close();

	    ofstream myfile_encoding;
	    myfile_encoding.open ("encoding_log.txt", ios::out | ios::app);
	    myfile_encoding << "\n";
	    myfile_encoding.close();
	} // end K loop

	ofstream myfile;
	myfile.open ("csr_log.txt", ios::out | ios::app);
	myfile << "\n";
	myfile.close();

	ofstream myfile_encoding;
	myfile_encoding.open ("encoding_log.txt", ios::out | ios::app);
	myfile_encoding << "\n";
	myfile_encoding.close();
   
 
   
	return 0;
}

