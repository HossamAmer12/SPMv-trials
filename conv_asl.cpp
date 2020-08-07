//
//  main.cpp
//  tf_conv
//
//  Created by Hossam Amer on 2020-05-10.
//  Copyright © 2020 Hossam Amer. All rights reserved.
//

// eigen_try1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
// Optm tricks: https://stackoverflow.com/questions/39547061/sparse-x-dense-matrix-multiplication-performance-under-efficient

#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
// #include <numeric>
// #include <tuple>
#include <iostream>
#include <vector>

#include <unsupported/Eigen/CXX11/Tensor>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

double diag = 3.0;
double off_diag = 1.7;
double off_off_diag = 1.5;


using namespace std;

/*
void print_matrix(SpMat &spm, const int mat_size) {
 //  Print the sparse matrix
 
  bool found_val = false;
  for (int i = 0; i < mat_size; i++) {
    for (int j = 0; j < mat_size; j++) {
      found_val = false;
      for (int k = 0; k < spm.size(); k++) {
        if (std::get<0>(spm[k]) == i && std::get<1>(spm[k]) == j) {
          printf("%8.4f\t", std::get<2>(spm[k]));
          found_val = true;
          break;
        }
      }
      if (found_val == false) {
        printf("%8.4f\t", 0.0);
      }
    }
    printf("\n");
  }
}
*/

void create_matrix(std::vector<T>& coeffs, SpMat& spm, const int mat_size) {
    /*
     Populate the sparse matrix for multiplication.
     */
    // TIMING
    auto start = std::chrono::high_resolution_clock::now();
    // TIMING
    
    // Popluating
    int reg_size = (int)mat_size / 3.0;
    
#pragma omp declare reduction( \
merge:std::vector<T>             \
: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : coeffs)
    for (int i = 1; i < reg_size; i++) {
        for (int j = 1; j < 5; j++) {
            // Region 1
            if (i * j < reg_size) {
                coeffs.push_back({i, i * j, log(i + j + 1.0)});
            }
            // Region 2
            if ((i + reg_size + j) % 2 == 0 && i + j < reg_size) {
                coeffs.push_back(
                                 {i + reg_size, i + reg_size + j, log(i + reg_size + j + 1)});
                coeffs.push_back(
                                 {i + reg_size + j, i + reg_size, log(i + reg_size + j + 1)});
            }
            // Region 3
            if (i + 2 * j < reg_size) {
                coeffs.push_back({i + reg_size, i + 2 * reg_size + 2 * j,
                    log(i + 2 * reg_size + j + 1)});
                coeffs.push_back({i + 2 * j + reg_size, i + 2 * reg_size,
                    log(i + 2 * reg_size + j + 1)});
                coeffs.push_back({i + 2 * reg_size, i + 2 * j + reg_size,
                    log(i + 2 * reg_size + j + 1)});
                coeffs.push_back({i + 2 * reg_size + 2 * j, i + reg_size,
                    log(i + 2 * reg_size + j + 1)});
            }
        }
    }
    
    //  A triplet is a simple object representing a non-zero entry as the triplet: row index, column index, value.
    // The raw and flat list of non-zero entries is then converted to a true SparseMatrix object A. 
    spm.setFromTriplets(coeffs.begin(), coeffs.end());
    // print_matrix(spm, mat_size);
    // exit(0);

    // TIMING
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	printf(
           "Eigen matrix population time %llu (milliseconds) for matrix of "
           "size %d x %d\n",
           duration.count(), mat_size, mat_size);
  
  // TIMING
}

void sparse_mv_mult(SpMat& spm, Eigen::VectorXd& b, Eigen::VectorXd& c,
                    const int mat_size) {
    /*
     Sparse matrix * dense vector mutliplication.
     */
    
    // TIMING
    auto start = std::chrono::high_resolution_clock::now();
    // TIMING
    
    c = spm * b;
    
    // TIMING
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    printf(
           "Eigen sparse matrix mult. time %llu (milliseconds) for matrix of size "
           "%d x "
           "%d\n",
           duration.count(), mat_size, mat_size);
    // TIMING
}


void create_matrix_den(std::vector<T>& coeffs, SpMat& spm, const int mat_size, const double density) {

   cout << "Required density is " << density << endl;

}

void im2col()
{
 cout << "---------------- START THE FUN " << endl;
 cout << "Mixed 0 - Conv Node 3 ru => " << 0.771 << endl;
 
 int padding = 0;
 int stride  = 1;
 int num_filters = 1; // 64
 
 // Input dimensions
/*
 int H = 4;
 int W = 4;
 int C = 3;
 int N = 1;
*/
 
 // mixed 0: conv_node 3
 int H = 149;
 int W = 149;
 int C = 32;
 int N = 1;
 
 // Kernel dimensions
/*
 int R = 2;
 int S = 2;
*/
 
 int R = 3;
 int S = 3;
 
 int Hout = (H - R + 2 * padding)/stride; // removed + 1
 int Wout = (W - S + 2 * padding)/stride;

 int kernel_dimX = R*S*C;
 int kernel_dimY = num_filters;

 int input_dimX = Hout * Wout;
 int input_dimY = R*S*C;
	
 cout << "Input Dim X: " << input_dimX << ", Input Dim Y: " << input_dimY << endl;
 cout << "Kernel Dim X: " << kernel_dimX << ", Kernel Dim Y: " << kernel_dimY << endl;

 // Create reshaped feature map
 Eigen::Tensor<double, 2> input(input_dimY, input_dimX);
 
 // Create reshaped Kernel
 Eigen::Tensor<double, 2> kernel(kernel_dimY, kernel_dimX);

 // Fill in some values in the input
for(int i = 0; i < input_dimY; ++i) {
   for(int j = 0; j < input_dimX; ++j) {
     input(i, j) = 1;
   }
 }
 
 // Fill in some values in the kernel
 for(int i = 0; i < kernel_dimY; ++i) {
   for(int j = 0; j < kernel_dimX; ++j) {
      kernel(i, j) = 1;
   }
 }
  
  // TIMING
  auto start = std::chrono::high_resolution_clock::now();
 
  // Compute the contraction for convolution:
  Eigen::array<Eigen::IndexPair<int>, 1> convolved_product_dims = { Eigen::IndexPair<int>(0, 1) };
  Eigen::Tensor<double, 2> conv_output  = input.contract(kernel, convolved_product_dims);
  // cout << conv_output << endl;

  const Eigen::Tensor<float, 2>::Dimensions& d = conv_output.dimensions();
  cout << "Size of Conv output  Yrange: " << d[0] << ", Xrange: " << d[1] << endl;

  // TIMING
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

  cout << "Dense Elapsed Time: " << duration.count() << " milliseconds" << endl;
}

int main(int argc, char** argv) {
    // Read command line options
    if (argc < 2) {
        printf(
               "Not enough arguments, use: \n\n./eigen_test N \n\nwhere N is the size "
               "of "
               "the "
               "matrix dimension\n");
        return -1;
    }
    const int mat_size = std::atoi(argv[1]);
    
    Eigen::initParallel();
    // Create a vector xd 
    Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(mat_size, 0, mat_size - 1);
    std::vector<T> coefficients;  // list of non-zeros coefficients
    SpMat spm(mat_size, mat_size);
    create_matrix(coefficients, spm, mat_size);
    
    Eigen::VectorXd c = Eigen::VectorXd::Zero(mat_size);
    // c.setZero();
    
    sparse_mv_mult(spm, b, c, mat_size);
    // std::cout << c << "\n";
    
    im2col();   
    // DEBUGGING
    // print_matrix(spm);
    // for (int i = 0; i < mat_size; i++) {
    //   printf("%f\n", b[i]);
    // }
    // printf("\n\n");
    // for (int i = 0; i < mat_size; i++) {
    //   printf("%f\n", c[i]);
    // }
    // DEBUGGING
    return 0;
}

