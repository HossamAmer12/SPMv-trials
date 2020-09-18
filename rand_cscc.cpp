#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <boost/timer/timer.hpp>
#include <time.h>
#include <fstream>


#define IS_PRINT        0
#define IS_PRINT_SIZE   0

using namespace Eigen;
using namespace std;
using namespace boost::timer;

// https://scicomp.stackexchange.com/questions/27977/how-can-i-speed-up-this-code-for-sparse-matrix-vector-multiplication

void bench_Sparse(const SparseMatrix<float> &m, const MatrixXf &in, MatrixXf &o) {
  // o.noalias() = m*in.transpose();
  o.noalias() = m*in;
}

void bench_Dense(const MatrixXf &m, const MatrixXf &in, MatrixXf &o) {
 // o.noalias() = m*in.transpose();

 // cout << m.rows() << ", " << m.cols() << endl;
 // cout << in.rows() << ", " << in.cols() << endl;
///  exit(0);
   o.noalias() = m*in;
//  o.noalias() = m.transpose()*in;

}

void print2DVectorF(std::vector<vector<float>>& x)
{
    for(int i = 0; i < x.size(); ++i)
    {   
        for(int j = 0; j < x[i].size(); ++j)
        {   
            cout << x[i][j] << " ";
        }   
        cout << "\n";
    }   
}

void printVector(std::vector<int>& x)
{
    cout << "\nPrint 1D Vector" << endl;
    for(int i = 0; i < x.size(); ++i)
    {
        cout << x[i] << ", ";
    }
    cout << "\n";
}

void printVector(std::vector<float>& x)
{
    cout << "\nPrint 1D Vector" << endl;
    for(int i = 0; i < x.size(); ++i)
    {
        cout << x[i] << ", ";
    }
    cout << "\n";
}

void printVector(std::vector<double>& x)
{
    cout << "\nPrint 1D Vector" << endl;
    for(int i = 0; i < x.size(); ++i)
    {
        cout << x[i] << ", ";
    }
    cout << "\n";
}




// CSR without eigen
void csrMult_v4(vector<vector<float> > & O, vector<int> const &K, vector<double> &Adata, vector<int> &Aindices, vector<int> &Aindptr, int Kh, int Kw, int Oh, int Ow)
{
  // cout << "Shape " << O.rows() << ", " << O.cols() << endl;
  
  int x                 = Aindptr[0];
  int *Aindex_help     = &Aindices[x];
  double *Adata_help   = &Adata[x];
  for (int n = 0; n < Ow; ++n)
  {
    for (; x < Aindptr[n + 1]; ++x)
    {   
      double result = 0.0;
      int NZE_index = *Aindex_help; Aindex_help++;
      int NZE_data  = *Adata_help; Adata_help++;

      if(x == Aindptr[n])
           cout << "*********ptr_type: " << 0 <<  " shereet: " << -1 << " i: "  << 0 << " submat: " << n << ", from: " << x << ", to: " << Aindptr[n + 1]<< endl;
        else
          cout << "ptr_type: " << 0 <<  " shereet: " << -1 << " i: "  << 0 << " submat: " << n << ", from: " << x << ", to: " << Aindptr[n + 1]<< endl;

      for(int l = 0; l < Kh; ++l)
      {   
        int m      = NZE_index/Kw - l;
        int Kindex = NZE_index%Kw + l*Kw; 
        if(m < 0 || m >= Oh) continue;
                 
        // cout << "R) " << m << ", C) " << n << ", " << Kindex << endl;
        if(m == 0 && n == 1)
        cout << "R) " << m << ", C) " << n << ", Data: " << Adata[x] << ", Index: " << Aindices[x] << endl;
        O[m][n] += NZE_data*K[Kindex];
      }   
    }   

  }
} // end mult


void csrMult_v1(MatrixXf& O, VectorXf& K, vector<double>& Adata, vector<int>& Aindices, vector<int>& Aindptr, int Kh, int Kw, int Oh, int Ow) 
{
  // cout << "Shape " << O.rows() << ", " << O.cols() << endl;
  
  int x = Aindptr[0]; 
  for (int n = 0; n < Ow; ++n)
  {
    for (; x < Aindptr[n + 1]; ++x)
    {   
      double result = 0.0;
      int NZE_index = Aindices[x];
      int NZE_data  = Adata[x];
      for(int l = 0; l < Kh; ++l)
      {   
        int m      = NZE_index/Kw - l;
        int Kindex = NZE_index%Kw + l*Kw; 
        if(m < 0 || m >= Oh) continue;
         
        // cout << "R) " << m << ", C) " << n << ", " << Kindex << endl;
        O(m, n) += NZE_data*K[Kindex];
      }   
    }   

  }
} // end mult

void csrMult_v2(MatrixXf& O, VectorXf& K, vector<double>& Adata, vector<int>& Aindices, vector<int>& Aindptr, int Kh, int Kw, int Oh, int Ow) 
{
  // cout << "Shape " << O.rows() << ", " << O.cols() << endl;
  
  int x                 = Aindptr[0];
  int *Aindex_help     = &Aindices[x];
  double *Adata_help   = &Adata[x];
  for (int n = 0; n < Ow; ++n)
  {
    for (; x < Aindptr[n + 1]; ++x)
    {   
      double result = 0.0;
      int NZE_index = *Aindex_help; Aindex_help++;
      int NZE_data  = *Adata_help; Adata_help++;
      for(int l = 0; l < Kh; ++l)
      {   
        int m      = NZE_index/Kw - l;
        int Kindex = NZE_index%Kw + l*Kw; 
        if(m < 0 || m >= Oh) continue;
                 
        // cout << "R) " << m << ", C) " << n << ", " << Kindex << endl;
        O(m, n) += NZE_data*K[Kindex];
      }   
    }   

  }
} // end mult

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
} // end mult


void csrMult(MatrixXf& O, VectorXf& K, vector<double>& Adata, vector<int>& Aindices, vector<int>& Aindptr, int Kh, int Kw, int Oh, int Ow)
{
  // cout << "Shape " << O.rows() << ", " << O.cols() << endl;
  
  for (int n = 0; n < Ow; ++n)
  {
    for (int x = Aindptr[n]; x < Aindptr[n + 1]; ++x)
    {
      for(int l = 0; l < Kh; ++l)
      {
        int m      = Aindices[x]/Kw - l;
        int Kindex = Aindices[x]%Kw + l*Kw; 
        if(m < 0 || m >= Oh) continue;
         
        // cout << "R) " << m << ", C) " << n << ", " << Kindex << endl;
        O(m, n) += Adata[x]*K[Kindex];
      }
    }

  }
} // end mult


int main()
{
  // density:
  // float density = 0.05;
  // float density = 1;
  float density = 0.5;
   
 // for(; density < 1.05; density+=0.05)
  {  


  // timer for im2col, csr
  float t_im2col = 0;
  float t_csr    = 0;
  float t_cpo    = 0;

  // bench iterations
//  int bench_iterations = 100000;
  // int bench_iterations = 2;
  int bench_iterations = 1;
  
  // Conv parameters:
  int padding = 0;
  int stride  = 1;
  int Sh, Sw; 
  Sh = Sw = stride;
  int num_filters = 1; // 64

  // mixed 0: conv_node 3
//  int Ih = 5;
//  int Iw = 5;
   
// int Ih = 149;
// int Iw = 149;

//  int Ih = 50;
//  int Iw = 50;
      
    // int Ih = 8;
    // int Iw = 8;

  int Ih = 17;
  int Iw = 17;


  // int Ih = 35;
  // int Iw = 35;

  // int Ih = 8;
  // int Iw = 8;
      
 
  // Kernel dimensions
  // int Kh = 3;
  // int Kw = 3;

  // int Kh = 7;
  // int Kw = 1;
  
  int Kh = 1;
  int Kw = 7;

  // int Kh = 5;
  // int Kw = 5;


  // adjust the iterations based on Ih  
  if(Ih > 100)
  {
    bench_iterations = 1000;
  }
  
  // int Ic = 32; // this is for the node
  int Ic = 1; // put it as articial for now
  int In = 1;

  int K = 1; // number of filters

  int Oh = (1 + Ih - Kh + 2 * padding)/stride; // removed + 1
  int Ow = (1 + Iw - Kw + 2 * padding)/stride;

  int iter = 1;  // total number of times to perform the test for each of dense, sparse multiplication

  // Create your original input feature map:
  MatrixXf org_fm = MatrixXf::Zero(Ih, Iw);

  std::vector<int> cols = {0,1,4,0,4,0,4};
  std::vector<int> rows = {0,0,0,2,2,3,3};
  std::vector<double> values = {1,1,1,1,1,1,1};
 
   
  for (int i = 0; i < ceil(density * Ih * Iw); ++i)
  {
    int r = rand() % Ih; 
    int c = rand() % Iw; 
    if(org_fm(r, c) == 0)
    {
       org_fm(r, c) = 1;
    }
    else
    {
      bool found = false;
      for(int u = 0; u < Ih; ++u)
      {
        for(int v = 0; v < Iw; ++v)
        {
          if(org_fm(u, v) == 0)
          {
            org_fm(u, v) = 1;
            found = true;
            break;
         }
       }
       if(found)
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
          if (org_fm(i, j) != 0){
            density_cal += 1;
          } 
      }
  }
  
  density_cal = density_cal/(Ih*Iw);
  // cout << "Calculated density: " << density_cal << endl;
  
  std::cout << "\n===Original Feature Map (" << Ih << "x" << Iw <<  "):  \n" << org_fm << std::endl;
  cout << "-----\n" << endl;
#if IS_PRINT
  // Print out the original feature map:
  std::cout << "\n===Original Feature Map (" << Ih << "x" << Iw <<  "):  \n" << org_fm << std::endl;
  cout << "-----\n" << endl;
#endif

#if IS_PRINT_SIZE
  // Print out the original feature map:
  std::cout << "\n===Original Feature Map (" << Ih << "x" << Iw <<  "):  \n" << std::endl;
  cout << "-----\n" << endl;
#endif  

  // Create the lowered matrix: 
  MatrixXf lowered_mat  = MatrixXf::Zero(Ow, Ih*Kw);
  int sub_matrix_index  = 0;  
  
  // For each submatrix:
  for (int sub_m_start_col = 0; sub_m_start_col < Ow; sub_m_start_col = sub_m_start_col + stride)
  {
   
   if(sub_m_start_col + Kw > Iw)
   { 
     break;
   }
   
   int lowered_mat_col_index = 0;
   // Fetch this piece (all rows, all cols in this current submatrix)
   for(int row_int = 0; row_int < Ih; ++row_int)
   {
     for(int col_int = sub_m_start_col; col_int < sub_m_start_col + Kw; ++col_int)
     {
         // cout << (org_fm(row_int, col_int)) << ", ";     
         
         lowered_mat(sub_matrix_index, lowered_mat_col_index) = org_fm(row_int, col_int);
         lowered_mat_col_index++;
     } // end inner loop
     
   } // end outer loop

   sub_matrix_index++;

  } // end outer outer loop

  std::cout << "\n===Lowered Feature Map of Size: " << lowered_mat.rows() << ", " << lowered_mat.cols() <<  "\n" << lowered_mat << std::endl;
  cout << "-----\n" << endl;   
#if IS_PRINT  
  // Print out the lowered feature map:
  std::cout << "\n===Lowered Feature Map of Size: " << lowered_mat.rows() << ", " << lowered_mat.cols() <<  "\n" << lowered_mat << std::endl;
  cout << "-----\n" << endl;   
#endif


#if IS_PRINT_SIZE  
  // Print out the lowered feature map:
  std::cout << "\n===Lowered Feature Map of Size: " << lowered_mat.rows() << ", " << lowered_mat.cols() <<  "\n"  << std::endl;
  cout << "-----\n" << endl;   
#endif

  int start_row_int = 0;
  int start_col_int = 0;

  // Create the sparse representation of the lowered matrix:
  SparseMatrix<float, RowMajor> lowered_mat_sparse = lowered_mat.sparseView();
  // SparseMatrix<int> lowered_mat_sparse = lowered_mat.sparseView();
  lowered_mat_sparse.makeCompressed();
 
#if IS_PRINT 
  // Print out the im2col interedmiate feature map:
  std::cout << "\n===CSR of Lowered Feature Map: " <<  " \n" << lowered_mat_sparse << std::endl;
  cout << "-----\n" << endl;
#endif  

  // Create the filter K and its vectorized version:
  MatrixXf filter             = MatrixXf::Ones(Kh, Kw);
  VectorXf filter_vectorized  = VectorXf::Ones(Kh*Kw);


#if IS_PRINT
  // Print out the im2col interedmiate feature map:
  std::cout << "\n===Filter: " <<  " \n" << filter_vectorized  << std::endl;
  cout << "-----\n" << endl;
#endif

#if IS_PRINT_SIZE
  // Print out the im2col interedmiate feature map:
  std::cout << "\n===Filter: " <<  " \n" << filter_vectorized  << std::endl;
  cout << "-----\n" << endl;
#endif

  // Prepare the output for im2col, sparseMat
  MatrixXf d_o1 = MatrixXf::Zero(Oh, Ow);
  MatrixXf d_o2 = MatrixXf::Zero(Oh, Ow);
  
  // Prepare the output for CPO
  vector<vector<float> > O( Oh , vector<float> (Ow, 0));

  // Create the Kernel
  vector<int> Kernel(Kh*Kw, 1);

  
  // Prepare the Adata, Aindices, AindPtr for CSR multiplication
  int nz = lowered_mat_sparse.nonZeros();
  vector<double> Adata (lowered_mat_sparse.valuePtr(), lowered_mat_sparse.valuePtr() + nz);
  vector<int> Aindices (lowered_mat_sparse.innerIndexPtr(), lowered_mat_sparse.innerIndexPtr() + nz);
  vector<int> Aindptr (lowered_mat_sparse.outerIndexPtr(), lowered_mat_sparse.outerIndexPtr() + lowered_mat_sparse.outerSize()); // +1 for the last element
  // push back the last element the number of nnz in ptr:
  Aindptr.push_back(nz);

  cout << "Ptr: " << endl;
  printVector(Aindptr);

  cout << "Index: " << endl;
  printVector(Aindices);

  cout << "Data: " << endl;
  printVector(Adata);




   // // Perform 50 times raw sparse matrix dense vector multiplication: d_o2 = d_m * d_b
   // {  
   //    clock_t t;
   //    t = clock(); 
   //    // for(int k=0;k<bench_iterations;k++) csrMult(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
   //    // for(int k=0;k<bench_iterations;k++) csrMult_v1(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
   //    // bench_iterations = 1; // if you want to see the correct result of csr_mult, comment this line
   //    for(int k=0;k<bench_iterations;k++) csrMult_v2(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
   //    double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds 
   //    t_csr+=elapsed/(Ih*Iw*1.0); // normalized timing
   // }

  // Perform 50 times raw sparse matrix dense vector multiplication: d_o2 = d_m * d_b [Without Eigen]
  {  
      clock_t t;
       for(int k=0;k<bench_iterations;k++){
           // Prepare the output for CSR
           vector<vector<float> > O_CSR( Oh , vector<float> (Ow, 0));
           t = clock();
           // csrMult_v2(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
           csrMult_v4(O_CSR, Kernel, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
           double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds 
           t_csr+=elapsed/(Ih*Iw*1.0); // normalized timing

           if(k == bench_iterations - 1)
           {
              cout << "CSR without Eigen Output: " << endl;
              print2DVectorF(O_CSR);
              cout << "-----\n" << endl;
           }
       }
   }

  //  // Print out the o1 from im2col:
  // std::cout << "\n===CSCC Output with Size: " << d_o2.rows() << ", " << d_o2.cols() <<  " \n" << d_o2 << std::endl;
  // cout << "-----\n" << endl;





  // elapsed time per feature element in the entire bench iterations
  // std::cout<<"batch\t"<<In<<"\tdensity\t"<<density<<"\tdensity\t"<<density_cal<<"\tim2col\t"<< t_im2col <<"\tcsr\t"<< t_csr <<"\tcpo\t"<< t_cpo <<std::endl;
  std::cout << "CSCC:\t" <<  Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batch\t"<<1
        <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
        <<"\tim2col\t"<<t_im2col<<"\tcsr\t" <<t_csr <<"\tcpo\t"<< t_cpo
         <<"\tpercent1\t"<< 100.0*(t_im2col-t_csr)/t_im2col  <<"\tpercent2\t"<< 100.0*(t_im2col-t_cpo)/t_im2col << "\n";

  
  
  } // density loop
 
  return 0;
}
