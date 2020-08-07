#include <iostream>
#include <vector>
#include <Eigen/Sparse>

using namespace Eigen;
using namespace std;

// https://scicomp.stackexchange.com/questions/27977/how-can-i-speed-up-this-code-for-sparse-matrix-vector-multiplication

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


int main()
{
  // Conv parameters:
  int padding = 0;
  int stride  = 1;
  int num_filters = 1; // 64

  // mixed 0: conv_node 3
  int Ih = 5;
  int Iw = 5;
  // int Ic = 32; // this is for the node
  int Ic = 1; // put it as articial for now

  int K = 1;

  // Kernel dimensions
  int Kh = 3;
  int Kw = 3;

  int Oh = (1 + Ih - Kh + 2 * padding)/stride; // removed + 1
  int Ow = (1 + Iw - Kw + 2 * padding)/stride;

  // Create your original input feature map:
  MatrixXf org_fm = MatrixXf::Zero(Ih, Iw);
  

  SparseMatrix<double, RowMajor> sm(Ih, Iw);
  // SparseMatrix<double, ColMajor> sm(4,5);

  std::vector<int> cols = {0,1,4,0,4,0,4};
  std::vector<int> rows = {0,0,0,2,2,3,3};
  // std::vector<double> values = {0.2,0.4,0.6,0.3,0.7,0.9,0.2};
  std::vector<double> values = {1,1,1,1,1,1,1};

  for(int i=0; i < cols.size(); i++)
  {
    sm.insert(rows[i], cols[i])     = values[i];
    org_fm(rows[i], cols[i])        = values[i];
  }
  
  sm.makeCompressed();

  // Prepare the Adata, Aindices, AindPtr for CSR multiplication
  int nz = sm.nonZeros();
  vector<double> Adata (sm.valuePtr(), sm.valuePtr() + nz);
  vector<int> Aindices (sm.innerIndexPtr(), sm.innerIndexPtr() + nz);
  vector<int> AindPtr (sm.outerIndexPtr(), sm.outerIndexPtr() + sm.outerSize()); // +1 for the last element
 
  // Print out the original feature map:
  std::cout << "\n===Original Feature Map: \n" << org_fm << std::endl;
  cout << "-----\n" << endl;
  
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
  
  // Print out the lowered feature map:
  std::cout << "\n===Lowered Feature Map of Size: " << lowered_mat.rows() << ", " << lowered_mat.cols() <<  "\n" << lowered_mat << std::endl;
  cout << "-----\n" << endl;   
  
  int start_row_int = 0;
  int start_col_int = 0;

  // Create the intermediate representation for im2col:
   MatrixXf im2col_mat    = MatrixXf::Zero(Oh*Ow, Kh*Kw*Ic);
  int patch_matrix_index = 0;
  
  // For each patch:
  for (int patch = 0; patch < Oh*Ow; patch = patch + stride)
  {

   int im2col_mat_col_index = 0;

   // Fetch this piece (all rows, all cols in this current submatrix)
   for(int row_int = start_row_int; row_int < start_row_int + Kh; ++row_int)
   {
     for(int col_int = start_col_int; col_int < start_col_int + Kw; ++col_int)
     {
	 // cout << "R) " << row_int << ", C) " << col_int << endl;
         // cout << (org_fm(row_int, col_int)) << ", ";     
         // cout << patch_matrix_index << ", " << im2col_mat_col_index << "=> " << im2col_mat.rows() << ", " << im2col_mat.cols() << endl;
         im2col_mat(patch_matrix_index, im2col_mat_col_index) = org_fm(row_int, col_int);
         im2col_mat_col_index++;
     } // end inner loop
    
   } // end outer loop
   
   patch_matrix_index++;
   
   // increment the start row
   start_row_int = start_row_int + stride;
   
   if(start_row_int + Kh > Ih)
   { 
    start_row_int = 0;
    start_col_int = start_col_int + stride;
   }
   
   
   if(start_col_int + Kw > Iw)
   {
     break;
   }

  } // end outer outer loop
  
  // Print out the im2col interedmiate feature map:
  std::cout << "\n===im2col Intermediate Feature Map with Size: " << im2col_mat.rows() << ", " << im2col_mat.cols() <<  " \n" << im2col_mat << std::endl;
  cout << "-----\n" << endl;
  
  // Create the sparse representation of the lowered matrix:
  SparseMatrix<float, RowMajor> lowered_mat_sparse = lowered_mat.sparseView();
  // SparseMatrix<int> lowered_mat_sparse = lowered_mat.sparseView();
  lowered_mat_sparse.makeCompressed();
  
  // Print out the im2col interedmiate feature map:
  std::cout << "\n===CSR of Lowered Feature Map: " <<  " \n" << lowered_mat_sparse << std::endl;
  cout << "-----\n" << endl;
  
  // Create the filter K 
  VectorXd filter  = VectorXd::Ones(Kh*Kw);

  // Print out the im2col interedmiate feature map:
  std::cout << "\n===Filter: " <<  " \n" << filter  << std::endl;
  cout << "-----\n" << endl;
  exit(0);



  std::cout << "\n===Sparse Feature Map: \n" << sm << std::endl;
  cout << "-----\n" << endl;
  std::cout << " Size of Ptr:  " << sm.outerSize() << endl;

  nz = sm.nonZeros();
  std::cout << "non_zeros : " << nz << " density: " << nz/(sm.size()*1.0) << std::endl;

  for (auto it = sm.valuePtr(); it != sm.valuePtr() + nz; ++it)
    std::cout << *it << std::endl;

  return 0;
}
