
// Disable bound assertions for C (dangerous)
#define NDEBUG

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <stdlib.h>
#include <boost/timer/timer.hpp>

#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>

#include <omp.h>
#include <vector>
#include <fstream>


// tutorial: https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
// https://stackoverflow.com/questions/57367167/for-eigen-sparsematrix-what-does-innerindexptr-and-outerindexptr-exactly-re

using namespace std;
using namespace Eigen;
using namespace boost::timer;

typedef SparseMatrix<float> spMatFloat;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> deMatRowFloat;


// https://scicomp.stackexchange.com/questions/27977/how-can-i-speed-up-this-code-for-sparse-matrix-vector-multiplication
// Ax = Adata * x
void csrMult_v3(VectorXd& Ax, VectorXd& x, vector<double>& Adata, vector<int>& Aindices, vector<int>& Aindptr)
{
   // This code assumes that the size of Ax is numRowsA.
   int dataIdx = Aindptr[0];
   int *Aindex = &Aindices[dataIdx];
 
 for(int j = 0; j < Aindptr.size(); ++j)
 {
  cout << j << ") " << Aindptr[j] << endl;
 }
  
   cout << "\nCaught the exception 22222 " << endl;
   cout << "value of i: " << 0 << ", Ax Size: " << Ax.size() << endl;
   cout << "Loop AindPtr "  << Aindptr[1] << ", AindPtr.size: " << Aindptr.size() << endl;
   cout << "dataIdx: " << dataIdx << endl;
   cout << "Adatat size "  << Adata.size() << endl;
  for (int i = 0; i < Ax.size(); i++)
  {
    double Ax_i = 0.0;
    for (; dataIdx < Aindptr[i + 1]; dataIdx++)
    {
      if(i > Aindptr.size())
      {
        cout << "Go beyond Aindptr.size "  << Aindptr.size() << ", " << Ax.size()  << endl;
        exit(0);
      } 
      if(dataIdx > Adata.size())
      { 
        cout << "\nCaught the exception " << endl;
        cout << "value of i: " << i << ", Ax Size: " << Ax.size() << endl;
        cout << "Loop AindPtr "  << Aindptr[i] << ", AindPtr.size: " << Aindptr.size() << endl;
        cout << "dataIdx: " << dataIdx << endl;
        cout << "Adatat size "  << Adata.size() << endl;
        exit(0);
      }
      Ax_i += Adata[dataIdx] * x[*Aindex];
    }

     Ax[i] = Ax_i;
  }
}


void bench_Sparse(const spMatFloat &m, const deMatRowFloat &in, deMatRowFloat &o) {
  o.noalias() = m*in.transpose();
}

void bench_Dense(const deMatRowFloat &m, const deMatRowFloat &in, deMatRowFloat &o) {
  o.noalias() = m*in.transpose();
}

int main(int argc, const char **argv) {
  
  float ratio=0.5; // density

  for(; ratio < 1.1; ratio+=0.1)
  { 
  // ***************** CONV PARAMETERS 
  int padding = 0;
  int stride  = 1;
  int num_filters = 1; // 64

  // mixed 0: conv_node 3
  int H = 149;
  int W = 149;
  int C = 32; 
  int N = 1;

  // Kernel dimensions
  int R = 3;
  int S = 3;

  int Hout = (H - R + 2 * padding)/stride; // removed + 1
  int Wout = (W - S + 2 * padding)/stride;

  int kernel_dimX = R*S*C;
  int kernel_dimY = num_filters;

  int input_dimX = Hout * Wout;
  int input_dimY = R*S*C;

  int input_lowered_dimX = Wout; // Wout
  int input_lowered_dimY = H*S; // Ih*Kw

  cout << "Input Dim X: " << input_dimX << ", Input Dim Y: " << input_dimY << endl;
  cout << "Kernel Dim X: " << kernel_dimX << ", Kernel Dim Y: " << kernel_dimY << endl;
  cout << "Output Dim X: " << Wout << ", Output Dim Y: " << Hout << endl;
 
 // ******************* CONV PARAMETERS
  int iter=20; // total number of times to perform the test for each of dense, sparse multiplication
  int batch=32; // batch size 
  float t_dense=0; // timer for dense multiplication 
  float t_sparse=0; // timer for sparse multiplication
  float t_csr=0; // timer for csr multiplication
  
  iter=1;  // assume that I do the computation once
  batch=1; // assume that batch size = 1 (more then you should use Eigen::Tensor)
 
 // ******************* Test PARAMETERS
   
  // deMatRowFloat d_o1(batch, Hout, Wout);
  // deMatRowFloat d_o2(batch, Hout, Wout);
 
  
  // Create the output matrices  
  // deMatRowFloat d_o1(batch, Hout*Wout);
  // deMatRowFloat d_o2(batch, Hout*Wout);
  deMatRowFloat d_o1 = deMatRowFloat::Zero(batch, Hout*Wout);
  deMatRowFloat d_o2 = deMatRowFloat::Zero(batch, Hout*Wout);
  VectorXd d_o3(batch*Hout*Wout);
  // cout << d_o2.rows() << ", " << d_o2.cols() << endl;
  // cout << d_o2.innerSize() << ", " << d_o2.outerSize() << endl;
  
  // In each iteration, compute the sparse matrix and dense matrix vector multiplications
  for(int k = 0; k < iter; ++k) {
     
    // Create the dense matrix (i.e intermediate matrix for im2col)
   // deMatRowFloat d_m = deMatRowFloat::Zero(input_dimY, input_dimX);
    deMatRowFloat d_m = deMatRowFloat::Zero(input_dimX, input_dimY);

    // Create the dense matrix (i.e lowered matrix for MEC)
    deMatRowFloat d_m_lowered = deMatRowFloat::Zero(input_lowered_dimX, input_lowered_dimY);
    
    // Create the filter (transpose in the product)
   // deMatRowFloat d_b = deMatRowFloat::Random(kernel_dimY, kernel_dimX);
   deMatRowFloat d_b = deMatRowFloat::Zero(kernel_dimY, kernel_dimX);
   // deMatRowFloat d_b = deMatRowFloat::Random(1, 1000);   
 
    // init some values for kernel: 
    for(int h = 0; h < ratio*kernel_dimY*kernel_dimX; ++h) {

      int i    = rand()%kernel_dimY;
      int j    = rand()%kernel_dimX;
      d_b(i, j) = 1;
    }
   
    // Vectorize the filter
    VectorXd d_b_vectorized  = VectorXd::Random(kernel_dimY*kernel_dimX);
    // VectorXd d_b_vectorized(Map<VectorXd>(d_b.data(), d_b.cols()*d_b.rows()));
    
    // fill our the dense matrix up to a certain density
    for(int h = 0; h < ratio*input_dimY*input_dimX; ++h) {

      int i    = rand()%input_dimY;
      int j    = rand()%input_dimX;
     // d_m(i, j) = (rand()%1000)/500.-1;
      // d_m(j, i) = (rand()%1000)/500.-1;	
      // d_m(j, i) = 12;
      d_m(j, i) = 1;
    }

    // Lowered matrix: 
    for(int h = 0; h < ratio*input_lowered_dimY*input_lowered_dimX; ++h) {

      int i    = rand()%input_lowered_dimY;
      int j    = rand()%input_lowered_dimX;
      d_m_lowered(j, i) = (rand()%1000)/500.-1;	
    }
   
    cout << "Performing product" << endl;
    cout << "LHS In ==> ( " << d_m.rows() << ", " << d_m.cols() << ")"  << endl;
    cout << "RHS Fil ==> ( " << d_b.transpose().rows() << ", " << d_b.transpose().cols() << ")" << endl;
    
    // Convert the dense matrix to Eigen sparse matrixu using sparse view
    spMatFloat s_m = d_m.sparseView();
     
    // Make compressed
    s_m.makeCompressed();

    // Create the CSR representation for lowered matrix
    spMatFloat s_m_lowered = d_m_lowered.sparseView();
    // Make compressed:
    s_m_lowered.makeCompressed();

    
    // Prepare the Adata, Aindices, AindPtr for CSR multiplication
    int nz = s_m_lowered.nonZeros();
    vector<double> Adata (s_m_lowered.valuePtr(),      s_m_lowered.valuePtr() + nz);
    vector<int> Aindices (s_m_lowered.innerIndexPtr(), s_m_lowered.innerIndexPtr() + nz);
    // vector<int> AindPtr (s_m.outerIndexPtr(), s_m.outerIndexPtr() + s_m.outerSize() + 1); // +1 for the last element
    vector<int> AindPtr (s_m_lowered.outerIndexPtr(), s_m_lowered.outerIndexPtr() + s_m_lowered.outerSize()); // +1 for the last element
    cout << s_m.outerSize() << ", " << AindPtr.size() << endl;
    // AindPtr[AindPtr.size()-1] = nz;
    
    // Perform 50 times sparse matrix dense multiplication using CSR algorithm: d_o3 = s_m * d_b_vectorized 
//    {
//      cpu_timer timer;
//      for(int k=0;k<50;k++) { csrMult_v3(d_o3, d_b_vectorized, Adata, Aindices, AindPtr);}
//      cpu_times const elapsed_times(timer.elapsed());
//      nanosecond_type const elapsed(elapsed_times.system+elapsed_times.user);
//      t_csr+=elapsed/(input_dimX*input_dimY * 1.0); // normalized timing
//    }

    // Perform 50 times dense matrix dense vector multiplication: d_o1 = d_m * d_b
    {
      cpu_timer timer;
      for(int k=0;k<50;k++) bench_Dense(d_m, d_b, d_o1);
      cpu_times const elapsed_times(timer.elapsed());
      nanosecond_type const elapsed(elapsed_times.system+elapsed_times.user);
      t_dense+=elapsed/(input_dimX*input_dimY * 1.0); // normalized timing
      
    }
    
    // Perform 50 times sparse matrix dense vector multiplication: d_o2 = s_m * d_b
    {
      cpu_timer timer;
      for(int k=0;k<50;k++) bench_Sparse(s_m,d_b,d_o2);
      cpu_times const elapsed_times(timer.elapsed());
      nanosecond_type const elapsed(elapsed_times.system+elapsed_times.user);
      t_sparse+=elapsed/(input_dimX*input_dimY * 1.0); // normalized timing
    }
  } // end total loop iterations
  
  cout << "O1 dims: " << "( " << d_o1.rows() << ", " << d_o1.cols() << ")" << " ==> O2 dims: " << "( " << d_o2.rows() << ", " << d_o2.cols() << ")" << " ==> O3 dims: " << "( " << d_o3.rows() << ", " << d_o3.cols() << ")" << endl;
  std::cout<<"\nbatch\t"<<batch<<"\tdensity\t"<<ratio<<"\tdense\t"<<t_dense/50/iter<<"\tsparse\t"<<t_sparse/50/iter <<"\tcsr\t"<<t_csr/50/iter<<std::endl;
  // bool r = d_o2.isApprox(d_o1); 
  // cout << r << endl;

  deMatRowFloat d_diff(batch, Hout*Wout);
  d_diff = d_o1 - d_o2;
  
  // cout << d_diff << endl;
  std::cout << " diff: " << d_diff.colwise().sum().sum() << std::endl;
   
  ofstream myfile;
  myfile.open ("example.txt", ios::out | ios::app);
  myfile << "\nbatch\t"<<batch<<"\tdensity\t"<<ratio<<"\tdense\t"<<t_dense/50/iter<<"\tsparse\t"<<t_sparse/50/iter<<"\tcsr\t"<<t_csr/50/iter;
  myfile.close();
 } 
} // end main
