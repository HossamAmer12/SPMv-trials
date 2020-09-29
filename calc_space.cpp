#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <boost/timer/timer.hpp>
#include <time.h>
#include <fstream>

// Set percision
#include <iomanip>



#define IS_PRINT        0
#define IS_PRINT_SIZE   0

using namespace Eigen;
using namespace std;
using namespace boost::timer;

// https://scicomp.stackexchange.com/questions/27977/how-can-i-speed-up-this-code-for-sparse-matrix-vector-multiplication

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

void print2DVector(std::vector<vector<int>>& x)
{
    for(int i = 0; i < x.size(); ++i)
    {
        cout << "\n===== " << i << " =====\n";
        for(int j = 0; j < x[i].size(); ++j)
        {
            cout << x[i][j] << ", ";
        }
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

/*SSSSSSSSSSSSSSSSSSSSSSSSSSS*/

void transform2dTo1dv1(vector<vector<int> >  &IN,  vector<vector<int> > &DA, vector<vector<int> >  &ptr, vector<int>  &IN_1d,  vector<int> &DA_1d, vector<int>  &ptr_1d)
{ 

          int c = 0;
         for(int i = 0 ; i < ptr.size(); ++i)
         {
          for(int j = 0; j < ptr[i].size(); ++j)
          {
            if(i == 0)
            {
              // Remove repeats
              if(j <= 1 || j == ptr[0].size() - 1)
              {
                  ptr_1d[c++] = ptr[i][j];
              }
              
            }
            else{
              ptr_1d[c++] = ptr[i][j];
            } 
            
          }
         } 
        
        int c1  = 0;
         for(int i = 0 ; i < DA.size(); ++i)
         {
          for(int j = 0; j < DA[i].size(); ++j)
          {
            DA_1d[c1] = DA[i][j];
            IN_1d[c1++] = IN[i][j];
          }
         }

}


void transform2dTo1d(vector<vector<int> >  &IN,  vector<vector<int> > &DA, vector<vector<int> >  &ptr, vector<int>  &IN_1d,  vector<int> &DA_1d, vector<int>  &ptr_1d)
{ 


          int c = 0;
         for(int i = 0 ; i < ptr.size(); ++i)
         {
          for(int j = 0; j < ptr[i].size(); ++j)
          {
            ptr_1d[c++] = ptr[i][j];
          }
         } 
        
        int c1  = 0;
         for(int i = 0 ; i < DA.size(); ++i)
         {
          for(int j = 0; j < DA[i].size(); ++j)
          {
            DA_1d[c1] = DA[i][j];
            IN_1d[c1++] = IN[i][j];
          }
         }

}

void transform2dTo1dv9(vector<vector<int> >  &IN,  vector<vector<int> > &DA, vector<vector<int> >  &ptr, vector<int>  &IN_1d,  vector<int> &DA_1d, vector<int>  &ptr_1d)
{ 

       int c = 0;
       for(int i = 0 ; i < ptr.size(); ++i)
       {
        for(int j = 0; j < ptr[i].size(); ++j)
        {
          // if(i == 0)
          // For all ptrs except the last one
          if(i <= ptr.size() - 2)
          {
            // Remove repeats
            if(j <= 1 || j == ptr[i].size() - 1)
            {
                ptr_1d[c++] = ptr[i][j];
            }
            
          }
          else{
            ptr_1d[c++] = ptr[i][j];
          } 
          
        }
       } 
      
       int c1  = 0;
       for(int i = 0 ; i < DA.size(); ++i)
       {
        for(int j = 0; j < DA[i].size(); ++j)
        {
          DA_1d[c1] = DA[i][j];
          IN_1d[c1++] = IN[i][j];
        }
       }

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


int main()
{
    
    
    // float density = 0.1;
//    float density = 0.5;

  // bench iterations
    int bench_iterations = 1;

    int DOUBLE_SIZE = 64;
   

 std::vector<int> I_list = {8,17,50};
  std::vector<int> Kh_list = {3,7, 1};
  std::vector<int> Kw_list = {3,1, 7};


  // std::vector<int> I_list = {8};
  // std::vector<int> Kh_list = {3};
  // std::vector<int> Kw_list = {3};

  // std::vector<int> I_list = {17};
  // std::vector<int> Kh_list = {3};
  // std::vector<int> Kw_list = {3};


  // std::vector<int> I_list = {17};
  // std::vector<int> Kh_list = {7};
  // std::vector<int> Kw_list = {1};

  // std::vector<int> I_list = {17};
  // std::vector<int> Kh_list = {1};
  // std::vector<int> Kw_list = {7};
    
    for(int KK = 0; KK < Kh_list.size(); ++KK)
    {
    
      for(int I: I_list)
      {
        int Ih = I;
        int Iw = I;

        // density:
     float density = 0.05;
    //    float density = 0.3;
  // float density = 0.1;
    

    for(; density < 1.05; density+=0.05)
    {
        
        // timer for im2col, csr
        float t_im2col = 0;
        float t_csr    = 0;
        float t_cpo    = 0;
        

        // Conv parameters:
        int padding = 0;
        int stride  = 1;
        int Sh, Sw;
        Sh = Sw = stride;
        int num_filters = 1; // 64
    

        
      int Kh = Kh_list[KK];
      int Kw = Kw_list[KK];
 
      // usleess case skip it
      if(Ih == 8)
        if(Kh == 1 || Kh == 7)
          continue;

        // int Ic = 32; // this is for the node
        int Ic = 1; // put it as articial for now
        int In = 1;
        
        int K = 1; // number of filters
        
        int Oh = (1 + Ih - Kh + 2 * padding)/stride; // removed + 1
        int Ow = (1 + Iw - Kw + 2 * padding)/stride;
        
        int iter = 1;  // total number of times to perform the test for each of dense, sparse multiplication
        
        // Create your original input feature map:
        MatrixXf org_fm = MatrixXf::Zero(Ih, Iw);
        
        
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


        // Space for im2col:
        t_im2col = im2col_mat.rows()*im2col_mat.cols()*DOUBLE_SIZE;

            
        // Prepare the Adata, Aindices, AindPtr for CSR multiplication
        // Create the sparse representation of the lowered matrix:
        SparseMatrix<float, RowMajor> lowered_mat_sparse = lowered_mat.sparseView();
        lowered_mat_sparse.makeCompressed();

        int nz = lowered_mat_sparse.nonZeros();
        vector<double> Adata (lowered_mat_sparse.valuePtr(), lowered_mat_sparse.valuePtr() + nz);
        vector<int> Aindices (lowered_mat_sparse.innerIndexPtr(), lowered_mat_sparse.innerIndexPtr() + nz);
        vector<int> Aindptr (lowered_mat_sparse.outerIndexPtr(), lowered_mat_sparse.outerIndexPtr() + lowered_mat_sparse.outerSize()); // +1 for the last element
        // push back the last element the number of nnz in ptr:
        Aindptr.push_back(nz);
    
        // cout << "creation cscc: " << t_cscc_creation << endl;
      

        // Space for CSCC:
        t_csr = (Adata.size() + Aindices.size() + Aindptr.size())*DOUBLE_SIZE;


        // ******************** CPO:
        int n = ceil(Kw / Sw);
        if (Kw % Sw == 0)
        {
            n = Kw / Sw;
        }
        
        vector<vector<int> > IN(n); // n is the rows
        vector<vector<int> > DA(n); // n is the rows
        vector<vector<int> > ptr( n , vector<int> (Ow + 1, 0)); // n is the rows
        CPO(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr);


        // V7 Code:
        // Remove repeats and transofrm 2d to 1d:
        int count_ptr_v7 = 0;
        for(int i = 0 ; i < ptr.size(); ++i)
        {

          // if it one ptr, don't do anything
          if(i == 0 && n != 1)
          {
            int f = ptr[i].size();
            count_ptr_v7 += min(f, 3);   
          }
          else
          {
            count_ptr_v7 += ptr[i].size();
          }
          
        } 

        int count_d_v7 = 0;
        for(int i = 0 ; i < DA.size(); ++i)
        {

          count_d_v7 += DA[i].size();
        }   

      std::vector<int> IN_1d_v7(count_d_v7, 0);
      std::vector<int> DA_1d_v7(count_d_v7, 0);
      std::vector<int> ptr_1d_v7(count_ptr_v7, 0);

      transform2dTo1dv1(IN, DA, ptr, IN_1d_v7, DA_1d_v7, ptr_1d_v7);
      


        if(n != 1)
        {
            // Space for CPO:  
            t_cpo = (ptr_1d_v7.size() + IN_1d_v7.size() + DA_1d_v7.size())*DOUBLE_SIZE;
      
        }
        else
        {
            // Space for CPO:  
            t_cpo = ( n * (1 + Ow) + IN_1d_v7.size() + DA_1d_v7.size())*DOUBLE_SIZE;      
        }

      
    
        bool s = (t_cpo <= t_csr);
        // elapsed time per feature element in the entire bench iterations
        //        std::cout<<"batch\t"<<In<<"\tdensity\t"<<density<<"\tdensity\t"<<density_cal<<"\tim2col\t"<< t_im2col <<"\tcsr\t"<< t_csr <<"\tcpo\t"<< t_cpo <<std::endl;
        
        std::cout << "B-" << bench_iterations << "\t" << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batch\t"<<1
        <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
        <<"\tim2col\t"<<t_im2col<<"\tcsr\t" <<t_csr <<"\tcpo\t"<< t_cpo 
        << "\tCR_CSCC\t"<< t_im2col/t_csr  <<"x\tCR_CPO\t"<< t_im2col/t_cpo << "\t" << s << "\n";

       
        
        ofstream myfile;
        myfile.open ("space_csr_log.txt", ios::out | ios::app);
        int batch = 1;
        
        // myfile << "B-" << bench_iterations << "\t" << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batch\t"<<1
        // <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
        // <<"\tim2col\t"<<t_im2col<<"\tcsr\t" <<t_csr <<"\tcpo\t"<< t_cpo 
        // << "\tCR_CSCC\t"<< t_im2col/t_csr  <<"x\tCR_CPO\t"<< t_im2col/t_cpo << "\t" << s << "\n";

        
        myfile << std::setprecision(3) << t_im2col/t_csr <<"\t" << t_im2col/t_cpo << "\n";

        

        
    } // density loop

    ofstream myfile;
    myfile.open ("space_csr_log.txt", ios::out | ios::app);
    myfile << "\n";
  } // end I loop

    ofstream myfile;
    myfile.open ("space_csr_log.txt", ios::out | ios::app);
    myfile << "\n";
} // end K loop
    
    return 0;
}

