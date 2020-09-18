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

/*SSSSSSSSSSSSSSSSSSSSSSSSSSS*/

void conv_CPO(vector<vector<float> > & O, vector<int> const &K, vector<vector<int> > const &IN,  vector<vector<int> > const &DA, vector<vector<int> > const &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;
    
    int n      = ceil(Kw/Sw);
    int number = floor((Iw - Kw)/Sw) + 1;
    // number = 0; n = 1;
    
    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    // For each ptr type
    // int type_ptr = 0;
    for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    {
    // For each submat
    for (int submat = 0; submat < number; ++submat)
    {
      // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " << ptr[type_ptr][submat] <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;
      
      // How many time to iterate?
      int shereet2 = min(submat, type_ptr);
      
      
      // int shereet2 = (submat == 0)? 0:type_ptr;
      // shereet2     = (submat == 1)? 1:shereet2; 
      for(int i = 0; i <= shereet2; ++i)
      {
  
       // From ptr r t r+1
        int shereet = (type_ptr > 0)? 1:0;

       for(int x = ptr[type_ptr][submat-i*shereet]; x < ptr[type_ptr][submat-i*shereet+1]; ++x)
      {

        // Loop on Kh for the output
       for(int l = 0; l < Kh; ++l)
       {
                    int input_index = IN[type_ptr][x] - i;
        int y_out = (input_index)/Kw - l;
        int x_out = submat;
        
        if(y_out < 0 || y_out >= Oh){
            // cout << "continue YYY============\n" << endl;
            continue;
         }
        
    
         // cout << "R) " << y_out << ", C) " << x_out << ", Data: " << DA[type_ptr][x] << ", Index: " << input_index  << ", ac_Index: " << IN[type_ptr][x] << endl;
    
        // O(y_out, x_out) += DA[type_ptr][x] * 1.0;
        O[y_out][x_out] += DA[type_ptr][x] * K[input_index%Kw + l*Kw];
      
       } // for each l in Kh

      } // for each raga3 el shereet
     } // for each x from range ptr and ptr + 1
  } // for each submat
  
    }

    // cout << "Output: " << endl;
    // print2DVectorF(O);
    // cout << "-----\n" << endl;
}





void CPO(MatrixXf& lowered_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> > &IN,
         vector<vector<int> > &DA, vector<vector<int> > &ptr)
{
    
    // std::cout << "\n===Lowered Feature Map of Size: " << lowered_mat.rows() << ", " << lowered_mat.cols() << "\n" << lowered_mat << std::endl;
    
    int flag = 0;
    int i = 0;
    int l = 0;
    int n;
    
    if (Kw % Sw == 0)
    {
        n = Kw / Sw;
    }
    else
    {
        n = ceil(Kw / Sw);
    }
    
    std::vector<int> x(n, 0);
    std::vector<int> m(n, 0);
    
    /*
     // Ptr declaration
     for (int p = 0; p < n; ++p)
     {
     ptr[p] = vector<int>(Ow + 1);
     }
     */
    
    // First piece
    for (int j = 0; j < Kw; ++j)
    {
        if (flag == 0)
        {
            ptr[l][m[l]] = x[l];
            flag = 1;
            m[l]++;
        } // end if (flag == 0)
        for (i = 0; i < Ih; ++i)
        {
            // cout << "I: " << i << " J: " << j << endl;
            
            if (lowered_mat(i, j) != 0)
            {
                // cout << i << ", " << j << endl;
                IN[l].push_back(j + (i * Kw));
                DA[l].push_back(lowered_mat(i, j));
                x[l]++;
            } // end if  if (lowered_mat(i, j) != 0)
            
        } // end for(i=0; i < Ih; ++i)
        
        if ((j + 1) % Sw == 0)
        {
            ptr[l][m[l]] = x[l];
            l++;
            flag = 0;
        } // end if ( (j+1) % Sw == 0)
        else if (j == Kw - 1)
        {
            ptr[l][m[l]] = x[l];
        } // end if(j == Kw - 1)
        
        // cout << "First piece: " << endl;
        // print2DVector(ptr);
        // printVector(m);
        
    } // end for (int j = 0; j < Kw; ++j)
    
    l--;
    
    for (int p = 0; p < m.size(); ++p)
    {
        m[p] = m[p] + 1;
        // printVector(m);
    }
    // printVector(m);
    flag = 0;
    
    
    // Second piece
    for (int j = Kw; j < Iw - Kw; ++j)
    {
        for (i = 0; i < Ih; ++i)
        {
            if (lowered_mat(i, j) != 0)
            {
                IN[l].push_back(j - ((m[l] - 1) * Sw) + (i * Kw));
                DA[l].push_back(lowered_mat(i, j));
                // IN[l][x[l]] = j - ((m[l] - 1) * Sw) + (i * Kw);
                // DA[l][x[l]] = lowered_mat(i, j);
                x[l]++;
            } // end if (lowered_mat(i, j) != 0)
        }// end for(i = 0; i < Ih; ++i)
        
        if (flag == 0)
        {
            if (Kw % Sw == 0)
            {
                if ((j - Kw + 1) % Sw == 0)
                {
                    ptr[l][m[l]] = x[l];
                    m[l]++;
                    
                    if (n > 1)
                    {
                        for (int c = 0; c < n - 1; ++c)
                        {
                            ptr[c][m[c]] = x[c];
                            m[c]++;
                        } // end for(int c = 0; c < n-1; ++c)
                    } // end if(n > 1)
                } // end if ((j - Kw + 1) % Sw == 0)
            } // end if (Kw % Sw == 0)
            else if ((j - Kw + 1) % Sw == (Sw - (Kw % Sw)))
            {
                ptr[l][m[l]] = x[l];
                m[l]++;
                l++;
                flag = 1;
            } // end else if ( (j - Kw + 1) % Sw == (Sw - (Kw % Sw)))
        } // end if(flag == 0)
        
        else
        {
            if ((j - Kw + 1) % Sw == 0)
            {
                ptr[l][m[l]] = x[l];
                m[l]++;
                l--;
                flag = 0;
                
            } // end if ( (j - Kw + 1) % Sw == 0)
            if (n > 2)
            {
                for (int c = 0; c < n - 2; ++c)
                {
                    ptr[c][m[c]] = x[c];
                    m[c]++;
                } // end for(int c = 0; c < n-2; ++c)
            } // end if n > 2
        } // end if flag == 1
        
    } // end for(int j = Kw; j < Iw - Kw; ++j)
    
    // printVector(m);
    
    // Third piece
    flag = 1;
    i = 0;
    for (int j = Iw - Kw; j < Iw; ++j)
    {
        for (int i = 0; i < Ih; i++)
        {
            // cout << "2222  Row: " << i << ", Col: " << j << endl;
            
            if (lowered_mat(i, j) != 0)
            {
                int ind_val = j + (i * Kw) - Sw * (m[l] - 1);
                // cout << "l " << l << " m[l]: " << m[l] << " Debug Index: " << ", j: " << j << ", (i*kw): " << (i * Kw) << " thir_part: " << (Sw * (m[l] - 1)) << " = " << ind_val << endl;
                //                << " 1 " << j + (i*Kw) - (Sw*(m[l] - 1)))<< endl;
                
                IN[l].push_back(ind_val);
                DA[l].push_back(lowered_mat(i, j));
                x[l]++;
            }// end if(lowered_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)
        
        if ((Iw - j - 1) % Sw == 0)
        {
            for (int c = 0; c < l + 1; ++c)
            {
                ptr[l][m[l]] = x[l];
                m[l]++;
            } // end for(int c = 0; c < l+1; ++c)
            
            if (l > 1)
            {
                for (int c = 0; c < l; ++c)
                {
                    ptr[c][m[c]] = x[c];
                    m[c]++;
                } // end for(int c = 0; c < l-1; ++c)
            }// end if l > 1
            
            else if (l == 1)
            {
                ptr[0][m[0]] = x[0];
                m[0]++;
            } // end else if(l == 1)
            
            l--;
        } // end if ((Iw - j - 1) % Sw == 0)
    } // end for (j = Iw - Kw; Iw; ++j)
    
    // cout << "\nPtr: ";
    // print2DVector(ptr);
    
    // cout << "\n\nIN: ";
    // print2DVector(IN);
    
    // cout << "\n\nData:";
    // print2DVector(DA);
    // cout << "\n" << endl;
    
}


/*SSSSSSSSSSSSSSSSSSSSSSSSSSS*/


// CSR without eigen
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


      for(int l = 0; l < Kh; ++l)
      {   
        int m      = NZE_index/Kw - l;
        int Kindex = NZE_index%Kw + l*Kw; 
        if(m < 0 || m >= Oh) continue;
                 
        // cout << "R) " << m << ", C) " << n << ", " << Kindex << endl;
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
     float density = 0.05;
    //    float density = 0.3;
    
//    float density = 0.5;
    
     for(; density < 1.05; density+=0.05)
    {
        
        // timer for im2col, csr
        float t_im2col = 0;
        float t_csr    = 0;
        float t_cpo    = 0;
        
        // bench iterations
        int bench_iterations = 100000;
        // int bench_iterations = 1;
        // int bench_iterations = 100;
        
        
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
        
        // int Ih = 50;
        // int Iw = 50;
        //       int Ih = 8;
         //      int Iw = 8;
        
         int Ih = 17;
         int Iw = 17;
        
        // Kernel dimensions
              // int Kh = 3;
              // int Kw = 3;
        
        int Kh = 7;
        int Kw = 1;
        
        // int Kh = 1;
        // int Kw = 7;
        
        
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
        
        // Create the intermediate representation for im2col:
        clock_t t_im2col_creation_c;
        t_im2col_creation_c = clock();


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

        double t_im2col_creation = 1000*((double)(clock()-t_im2col_creation_c))/CLOCKS_PER_SEC; // time in milliseconds
        t_im2col_creation = t_im2col_creation/(Ih*Iw*1.0); // normalized timing
        // cout << "creation im2col: " << t_im2col_creation << endl;
        
#if IS_PRINT
        // Print out the im2col interedmiate feature map:
        std::cout << "\n===im2col Intermediate Feature Map with Size: " << im2col_mat.rows() << ", " << im2col_mat.cols() <<  " \n" << im2col_mat << std::endl;
        cout << "-----\n" << endl;
#endif
        
        
#if IS_PRINT_SIZE
        // Print out the im2col interedmiate feature map:
        std::cout << "\n===im2col Intermediate Feature Map with Size: " << im2col_mat.rows() << ", " << im2col_mat.cols() <<  " \n" << std::endl;
        cout << "-----\n" << endl;
#endif
        
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
        
        // transpose the matrix for im2col:
        MatrixXf im2col_mat_tr = im2col_mat.transpose();
        
        // Perform 50 times dense matrix dense vector multiplication: d_o1 = d_m * d_b
        {
            clock_t t;
            t = clock();
            for(int k=0;k<bench_iterations;k++)  bench_Dense(im2col_mat, filter_vectorized, d_o1);
            double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
            t_im2col+=elapsed/(Ih*Iw*1.0); // normalized timing

            // include creation time:
            // t_im2col +=  t_im2col_creation;

        }
        
        
#if IS_PRINT
        // Print out the o1 from im2col:
        std::cout << "\n===im2col Output with Size: " << d_o1.rows() << ", " << d_o1.cols() <<  " \n" << d_o1 << std::endl;
        cout << "-----\n" << endl;
#endif
        
        
#if IS_PRINT_SIZE
        // Print out the o1 from im2col:
        std::cout << "\n===im2col Output with Size: " << d_o1.rows() << ", " << d_o1.cols() << std::endl;
        cout << "-----\n" << endl;
#endif
        
        // Prepare the Adata, Aindices, AindPtr for CSR multiplication
        clock_t t_cscc_creation_c;
        t_cscc_creation_c = clock();

        int nz = lowered_mat_sparse.nonZeros();
        vector<double> Adata (lowered_mat_sparse.valuePtr(), lowered_mat_sparse.valuePtr() + nz);
        vector<int> Aindices (lowered_mat_sparse.innerIndexPtr(), lowered_mat_sparse.innerIndexPtr() + nz);
        vector<int> Aindptr (lowered_mat_sparse.outerIndexPtr(), lowered_mat_sparse.outerIndexPtr() + lowered_mat_sparse.outerSize()); // +1 for the last element
        // push back the last element the number of nnz in ptr:
        Aindptr.push_back(nz);
    
        double t_cscc_creation = 1000*((double)(clock()-t_cscc_creation_c))/CLOCKS_PER_SEC; // time in milliseconds
        t_cscc_creation = t_cscc_creation/(Ih*Iw*1.0); // normalized timing        
        // cout << "creation cscc: " << t_cscc_creation << endl;

        // With Eigen
               // Perform 50 times raw sparse matrix dense vector multiplication: d_o2 = d_m * d_b
               // {
               //     clock_t t;
               //     t = clock();
               //     // for(int k=0;k<bench_iterations;k++) csrMult(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
               //     // for(int k=0;k<bench_iterations;k++) csrMult_v1(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
               //     // bench_iterations = 1; // if you want to see the correct result of csr_mult, comment this line
               //     for(int k=0;k<bench_iterations;k++) csrMult_v2(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
               //     double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
               //     t_csr+=elapsed/(Ih*Iw*1.0); // normalized timing
               // }
        
        // Without Eigen:
        // Perform 50 times raw sparse matrix dense vector multiplication: d_o2 = d_m * d_b [Without Eigen]
        {
            // Prepare the output for CSCC
            vector<vector<float> > O_CSR( Oh , vector<float> (Ow, 0));
            
            clock_t t;
            t = clock();
            for(int k=0;k<bench_iterations;k++){
                csrMult_v4(O_CSR, Kernel, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
//                if(k == bench_iterations-1)
//                {
//                    cout << "CSR without Eigen Output: " << endl;
//                    print2DVectorF(O_CSR);
//                    cout << "-----\n" << endl;
//                }
            } // end k lop 
            double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
            t_csr += elapsed/(Ih*Iw*1.0); // normalized timing

        
                // include creation time:
               // t_csr +=  t_cscc_creation;
        }
        
        
#if IS_PRINT
        // Print out the o1 from im2col:
        std::cout << "\n===CSCC Output with Size: " << d_o2.rows() << ", " << d_o2.cols() <<  " \n" << d_o2 << std::endl;
        cout << "-----\n" << endl;
#endif
        
        
#if IS_PRINT_SIZE
        // Print out the o1 from im2col:
        std::cout << "\n===CSCC Output with Size: " << d_o2.rows() << ", " << d_o2.cols() << std::endl;
        cout << "-----\n" << endl;
#endif
        
        // Prepare the Adata, Aindices, AindPtr for CPO multiplication
        clock_t t_cpo_creation_c;
        t_cpo_creation_c = clock();

        int n = ceil(Kw / Sw);
        if (Kw % Sw == 0)
        {
            n = Kw / Sw;
        }
        
        vector<vector<int> > IN(n); // n is the rows
        vector<vector<int> > DA(n); // n is the rows
        vector<vector<int> > ptr( n , vector<int> (Ow + 1, 0)); // n is the rows
        CPO(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr);

        double t_cpo_creation = 1000*((double)(clock()-t_cpo_creation_c))/CLOCKS_PER_SEC; // time in milliseconds
        t_cpo_creation = t_cpo_creation/(Ih*Iw*1.0); // normalized timing
        // cout << "creation cpo: " << t_cpo_creation << endl;

        
        // Perform 50 times raw sparse matrix dense vector multiplication: d_CPO = d_m * d_b
        {
            // Prepare the output for CPO
            vector<vector<float> > O( Oh , vector<float> (Ow, 0));
            
            clock_t t;
            t = clock();
            for(int k=0;k<bench_iterations;k++){
                conv_CPO(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
//                if(k == bench_iterations-1)
//                {
//                    print2DVectorF(O);
//                    cout << "-----\n" << endl;
//                }
            } // end k lop 
            double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
            t_cpo+=elapsed/(Ih*Iw*1.0); // normalized timing

        
                // include creation time:
               // t_cpo += t_cpo_creation;
        }
                
        // elapsed time per feature element in the entire bench iterations
        //        std::cout<<"batch\t"<<In<<"\tdensity\t"<<density<<"\tdensity\t"<<density_cal<<"\tim2col\t"<< t_im2col <<"\tcsr\t"<< t_csr <<"\tcpo\t"<< t_cpo <<std::endl;
        std::cout << "B-" << bench_iterations << "\t" << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batch\t"<<1
        <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
        <<"\tim2col\t"<<t_im2col<<"\tcsr\t" <<t_csr <<"\tcpo\t"<< t_cpo
        <<"\tpercent1\t"<< 100.0*(t_im2col-t_csr)/t_im2col  <<"\tpercent2\t"<< 100.0*(t_im2col-t_cpo)/t_im2col << "\n";
        
        //        ofstream myfile;
        //        myfile.open ("csr_log.txt", ios::out | ios::app);
        //        int batch = 1;
        //        myfile << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batch\t"<<batch
        //        <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
        //        <<"\tim2col\t"<<t_im2col<<"\tcsr\t" <<t_csr <<"\tcpo\t"<< t_cpo
        //         <<"\tpercent1\t"<< 100.0*(t_im2col-t_csr)/t_im2col  <<"\tpercent2\t"<< 100.0*(t_im2col-t_cpo)/t_im2col << "\n";
        //        myfile.close();
        
    } // density loop
    
    return 0;
}
