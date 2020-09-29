#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
//#include <boost/timer/timer.hpp>
#include <time.h>
#include <math.h>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

using namespace Eigen;
using namespace std;
//using namespace boost::timer;

void printVectorD(std::vector<double>& x)
{
    cout << "\nPrint 1D Vector" << endl;
    for(int i = 0; i < x.size(); ++i)
    {
        cout << x[i] << ", ";
    }
    cout << "\n";
}



void print2DVector(std::vector<vector<int>>& x)
{
    for (int i = 0; i < x.size(); ++i)
    {
        std::cout << "\n===== ROW :" << i << " =====\n";
        for (int j = 0; j < x[i].size(); ++j)
        {
            std::cout << x[i][j] << ", ";
        }
    }
}

void printVector(std::vector<int>& x)
{
    std::cout << "\nPrint 1D Vector" << endl;
    for (int i = 0; i < x.size(); ++i)
    {
        std::cout << x[i] << ", ";
    }
    std::cout << "\n";
}


void CPO1(MatrixXf& O, VectorXf& K, MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
    int flag = 0;
    int i = 0;
    int l = 0;
    int n = ceil(Kw / Sw);

    std::vector<int> x(n, 0);
    std::vector<int> m(n, 0);
    vector<vector<int> > IN(n); // n is the rows
    vector<vector<int> > DA(n); // n is the rows
    vector<vector<int> > ptr(n); // n is the rows

    // Ptr declaration
    for (int p = 0; p < n; ++p)
    {
        ptr[p] = vector<int>(Ow + 1);
    }

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
            if (org_mat(i, j) != 0)
            {
                IN[l].push_back(j + (i * Kw));
                DA[l].push_back(org_mat(i, j));
                x[l]++;
            } // end if  if (org_mat(i, j) != 0)
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
    } // end for (int j = 0; j < Kw; ++j)

    l--;

    for (int p = 0; p < m.size(); ++p)
    {
        m[p] = m[p] + 1;
    }
    flag = 0;

    // Second piece
    for (int j = Kw; j < Iw - Kw; ++j)
    {
        for (i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                int ind_val = j + (i * Kw) - Sw * (m[l] - 1);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
            } // end if (org_mat(i, j) != 0)
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

    // Third piece
    for (int j = Iw - Kw; j < Iw; ++j)
    {
        for (int i = 0; i < Ih; i++)
        {
            if (org_mat(i, j) != 0)
            {
                int ind_val = j + (i * Kw) - Sw * (m[l] - 1);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
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

    // std::cout << "\nPtr: ";
   //  print2DVector(ptr);

    // std::cout << "\n\nIN: ";
     //print2DVector(IN);

  //   std::cout << "\n\nData:";
  //   print2DVector(DA);
   //  std::cout << "\n" << endl;
}

void CPO2(MatrixXf& O, VectorXf& K, MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{

    int l = 0;
    int n = Kw;

    std::vector<int> x(n, 0);
    std::vector<int> m(n, 0);
    vector<vector<int> > IN(n); // n is the rows
    vector<vector<int> > DA(n); // n is the rows
    vector<vector<int> > ptr(n); // n is the rows

    // Ptr declaration
    for (int p = 0; p < n; ++p)
    {
        ptr[p] = vector<int>(Ow + 1);
    }

    // First part
    for (int j = 0; j < Kw; ++j)
    {
        ptr[l][m[l]] = x[l];
        m[l]++;
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                int ind_val = j + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)       
        ptr[l][m[l]] = x[l];
        m[l]++;
        l++;
    } // end for (int j = 0; j < Kw; ++j)

    l--;

    // Second piece
    for (int j = Kw; j < Iw - Kw; ++j)
    {
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                int ind_val = j - (m[l] - 1) + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
            } // end if (org_mat(i, j) != 0)
        }// end for(i = 0; i < Ih; ++i)
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
    } // end for(int j = Kw; j < Iw - Kw; ++j)

    // Third piece   
    for (int j = Iw - Kw; j < Iw; ++j)
    {
        for (int i = 0; i < Ih; i++)
        {
            if (org_mat(i, j) != 0)
            {
                int ind_val = j - (m[l] - 1) + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
            }// end if(org_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)
        for (int c = 0; c < l + 1; ++c)
        {
            ptr[l][m[l]] = x[l];
            m[l]++;
        } // end for(int c = 0; c < l+1; ++c)
        if (l >= 1)
        {
            for (int c = 0; c < l; ++c)
            {
                ptr[c][m[c]] = x[c];
                m[c]++;
            } // end for(int c = 0; c < l-1; ++c)
        }// end if l > 1
        l--;
    } // end for (j = Iw - Kw; Iw; ++j)

    //std::cout << "\nPtr: ";
   // print2DVector(ptr);

    //std::cout << "\n\nIN: ";
    //print2DVector(IN);

    // std::cout << "\n\nData:";
     // print2DVector(DA);
    //std::cout << "\n" << endl;
}

void CPO3(MatrixXf& O, VectorXf& K, MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{

    int l = 0;
    int n = Kw;

    std::vector<int> x(n, 0);
    std::vector<int> m(n, 0);
    vector<vector<int> > IN(n); // n is the rows
    vector<vector<int> > DA(n); // n is the rows
    vector<vector<int> > ptr(n); // n is the rows

    // Ptr declaration
    for (int p = 0; p < n; ++p)
    {
        ptr[p] = vector<int>(Ow + 1);
    }

    // First part
    for (int j = 0; j < Kw; ++j)
    {
        ptr[l][m[l]] = x[l];
        m[l]++;
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                int ind_val = j + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)       
        ptr[l][m[l]] = x[l];
        m[l]++;
        l++;
    } // end for (int j = 0; j < Kw; ++j)

    l--;

    // Second piece
    for (int j = Kw; j < Iw - Kw; ++j)
    {
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                int ind_val = j - (m[l] - 1) + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
            } // end if (org_mat(i, j) != 0)
        }// end for(i = 0; i < Ih; ++i)
        ptr[l][m[l]] = x[l];
        m[l]++;
    } // end for(int j = Kw; j < Iw - Kw; ++j)

    // Third piece   
    for (int j = Iw - Kw; j < Iw; ++j)
    {
        for (int i = 0; i < Ih; i++)
        {
            if (org_mat(i, j) != 0)
            {
                int ind_val = j - (m[l] - 1) + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
            }// end if(org_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)
        for (int c = 0; c < l + 1; ++c)
        {
            ptr[l][m[l]] = x[l];
            m[l]++;
        } // end for(int c = 0; c < l+1; ++c)
        l--;
    } // end for (j = Iw - Kw; Iw; ++j)

    //std::cout << "\nPtr: ";
   // print2DVector(ptr);

   // std::cout << "\n\nIN: ";
  //  print2DVector(IN);

    // std::cout << "\n\nData:";
     // print2DVector(DA);
   // std::cout << "\n" << endl;
}

void _CPO(MatrixXf& O, VectorXf& K, MatrixXf& org_fm, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
    int i = 0;
    int l = 0;
    int n = ceil(Kw / Sw);
    int count = 0;
    int subMatrix_count = 0;
    std::vector<int> numCols_ptr;
    std::vector<int> ptr_indexType;
    std::vector<int> subMatrix_index;
    // we can fix a specific place in the memory for each value
    for (int c = 0; c < floor(Kw / Sw); ++c)
    {
        numCols_ptr.push_back(Sw);
        ptr_indexType.push_back(count);
        subMatrix_index.push_back(subMatrix_count);
        count++;
    }
    if ((Kw % Sw) != 0)
    {
        numCols_ptr.push_back(Kw % Sw);
        ptr_indexType.push_back(count);
        subMatrix_index.push_back(subMatrix_count);
        count++;
    }

    subMatrix_count++;
    for (int c = 0; c < ceil((Iw - (2 * Kw)) / Sw); ++c)
    {
        numCols_ptr.push_back(Sw - (Kw % Sw));
        numCols_ptr.push_back(Kw % Sw);

        ptr_indexType.push_back(floor(Kw / Sw) - 1);
        ptr_indexType.push_back(floor(Kw / Sw));

        subMatrix_index.push_back(subMatrix_count);
        subMatrix_index.push_back(subMatrix_count);
        count += 2;
        subMatrix_count++;
    }

    for (int c = 0; c < int(floor(Kw / Sw)); ++c)
    {
        numCols_ptr.push_back(Sw);
        ptr_indexType.push_back(floor(Kw / Sw) - (c + 1));
        subMatrix_index.push_back(subMatrix_count);
        count++;
        subMatrix_count++;
    }

    int j = 0; int num_nnz = 0;
    vector<vector<int> > Ptr(n); // n is the rows
    vector<vector<int> > DN(n); // n is the rows
    vector<vector<int> > IN(n); // n is the rows
    int tmp;
    // Ptr declaration
    for (int p = 0; p < n; ++p)
    {
        Ptr[p] = vector<int>(Ow + 1, 0);
    }

    for (int i_ = 0; i_ < numCols_ptr.size(); ++i_)
    {
        if (numCols_ptr[i_] != 0)
        {
            for (int j_ = 0; j_ < numCols_ptr[i_]; ++j_)
            {
                for (int i = 0; i < Ih; ++i)
                {
                    if (org_fm(i, j) != 0)
                    {
                        IN[ptr_indexType[i_]].push_back((j + (i * Kw)) - (subMatrix_index[i_] * Sw));
                        DN[ptr_indexType[i_]].push_back(org_fm(i, j));
                        num_nnz++;
                    }
                } // end loop 1
                j++;
            }// end loop 2
            tmp = subMatrix_index[i_];
            for (int ptr_c = 0; ptr_c < n; ++ptr_c)
            {
                Ptr[ptr_c][tmp + 1] += Ptr[ptr_c][tmp];
            }
            Ptr[ptr_indexType[i_]][tmp + 1] = num_nnz + Ptr[ptr_indexType[i_]][tmp + 1];
        }
        num_nnz = 0;
    }// end loop 3
   // std::cout << "Pointers\n";
   // print2DVector(Ptr);
   // std::cout << "\nIndincies\n";
   // print2DVector(IN);
    //  std::cout << "\nData\n";
     // print2DVector(DN);
}

void CSR1(MatrixXf& O, VectorXf& K, MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
    int i = 0;
    int x = 0;
    int m = 0;

    std::vector<int> IN;
    std::vector<int> DA;
    std::vector<int> ptr;

    ptr.push_back(x);
    for (int j = 0; j < Iw; ++j)
    {
        if (org_mat(i, j) != 0)
        {
            IN.push_back(j + (i * Kw) - m);
            DA.push_back(org_mat(i, j));
            x++;
        } // end if (org_mat(i, j) != 0)
        if (j == m + (Kw - 1))
        {
            i++;
            j = m - 1;
            if (i == Ih)
            {
                i = 0;
                j = m - 1;
                m++;
                ptr.push_back(x);
            } // end if (i == Ih)
        } // end if (j == m + (Kw - 1))
    } // end for (int j = 0; j < Iw; ++j)

    std::cout << "\nPtr: ";
    printVector(ptr);

    std::cout << "\n\nIN: ";
    printVector(IN);

    std::cout << "\n\nData:";
    printVector(DA);
    std::cout << "\n" << endl;
}

void CSR2(MatrixXf& O, VectorXf& K, MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
    int x = 0;
    int m = 0;

    std::vector<int> IN;
    std::vector<int> DA;
    std::vector<int> ptr;

    ptr.push_back(x);
    for (int j = 0; j < Iw; ++j)
    {
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                IN.push_back(j + (i * Kw) - m);
                DA.push_back(org_mat(i, j));
                x++;
            } // end if (org_mat(i, j) != 0)          
        } // end for(int i = 0; i < Ih; ++i)       
        if (j == m + (Kw - 1))
        {
            if (j == Iw - 1)
            {
                break;
            }
            m++;
            j = m - 1;
            ptr.push_back(x);

        } // end if (j == m + (Kw - 1)) 

    } // end for (int j = 0; j < Iw; ++j)


    std::cout << "\nPtr: ";
    printVector(ptr);

    std::cout << "\n\nIN: ";
    printVector(IN);

    std::cout << "\n\nData:";
    printVector(DA);
    std::cout << "\n" << endl;
}
void Im2col(MatrixXf& O, VectorXf& K, MatrixXf& org_fm, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, int Ic)
{
    // Prepare the output for im2col, sparseMat, CPO
    MatrixXf d_o1 = MatrixXf::Zero(Oh, Ow);
    MatrixXf d_o2 = MatrixXf::Zero(Oh, Ow);
    MatrixXf d_o3 = MatrixXf::Zero(Oh, Ow);
    double elapsed_1, elapsed_2, elapsed_3;
    // Perform 50 times raw sparse matrix dense vector multiplication of CPO: d_o3 = d_m * d_b
    int start_row_int = 0;
    int start_col_int = 0;
    // Create the intermediate representation for im2col:
    MatrixXf im2col_mat = MatrixXf::Zero(Oh * Ow, Kh * Kw * Ic);
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


int main()
{
    // density:
    float density = 0.1;

    // timer for im2col, csr
    float t_im2col = 0;
    float t_csr = 0;

    // bench iterations
    int bench_iterations = 1;


    // Conv parameters:
    int padding = 0;
    int stride = 1;
    int Sh, Sw;
    Sh = Sw = stride;
    int num_filters = 1; // 64

    // mixed 0: conv_node 3
    int Ih, Iw;
    Ih = Iw = 8;

    int Ic = 1; // put it as articial for now
    int In = 1;

    int K = 1; // number of filters

    // Kernel dimensions
    int Kh;
    int Kw;
    Kh = 3;
    Kw = 3;

    int Oh = (1 + Ih - Kh + 2 * padding) / stride; // removed + 1
    int Ow = (1 + Iw - Kw + 2 * padding) / stride;

    int iter = 1;  // total number of times to perform the test for each of dense, sparse multiplication

    // Create your original input feature map:
    MatrixXf org_fm = MatrixXf::Zero(Ih, Iw);

    org_fm(0, 6) = 1;
    org_fm(1, 2) = 1;
    org_fm(2, 0) = 1;
    org_fm(7, 1) = 1;
    //org_fm(1, 2) = 2;
    //org_fm(4, 2) = 1;
    //org_fm(1, 3) = 1;
    //org_fm(2, 3) = 1;
    //org_fm(3, 3) = 1;
    //org_fm(4, 3) = 1;
    //org_fm(5, 3) = 1;
   // org_fm(1, 4) = 1;
    //org_fm(2, 4) = 1;
   // org_fm(3, 4) = 2;
    //org_fm(4, 4) = 1;
    //org_fm(1, 5) = 2;
    //org_fm(4, 5) = 1;
   // org_fm(5, 5) = 8;
   // org_fm(0, 3) = 1;
   // org_fm(0, 4) = 3;
   // org_fm(2, 0) = 1;
    //org_fm(3, 0) = 2;
    //org_fm(1, 6) = 5;



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

    // Create the sparse representation of the lowered matrix:
    SparseMatrix<float, RowMajor> lowered_mat_sparse = lowered_mat.sparseView();

    // SparseMatrix<int> lowered_mat_sparse = lowered_mat.sparseView();
    lowered_mat_sparse.makeCompressed();


    int nz = lowered_mat_sparse.nonZeros();
    vector<double> Adata (lowered_mat_sparse.valuePtr(), lowered_mat_sparse.valuePtr() + nz);
    vector<int> Aindices (lowered_mat_sparse.innerIndexPtr(), lowered_mat_sparse.innerIndexPtr() + nz);
    vector<int> Aindptr (lowered_mat_sparse.outerIndexPtr(), lowered_mat_sparse.outerIndexPtr() + lowered_mat_sparse.outerSize()); // +1 for the last element
    // push back the last element the number of nnz in ptr:
    Aindptr.push_back(nz);

    cout << "Eigen CSR: " << endl;
    cout << "Data: "; 
    printVectorD(Adata);
    cout << "Indices: ";
    printVector(Aindices);
    cout << "ptr: ";
    printVector(Aindptr);


    // Create the filter K and its vectorized version:
    MatrixXf filter = MatrixXf::Ones(Kh, Kw);
    VectorXf filter_vectorized = VectorXf::Ones(Kh * Kw);


    // Prepare the output for im2col, sparseMat, CPO
    MatrixXf d_o1 = MatrixXf::Zero(Oh, Ow);
    MatrixXf d_o2 = MatrixXf::Zero(Oh, Ow);
    MatrixXf d_o3 = MatrixXf::Zero(Oh, Ow);


    double elapsed_1, elapsed_2, elapsed_3, elapsed_4, elapsed_5, elapsed_6, elapsed_7;

    {
        clock_t t;
        t = clock();
        // auto t1 = Clock::now();
        for (int k = 0; k < bench_iterations; k++) CPO1(d_o3, filter_vectorized, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
        elapsed_1 = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds
        // auto t2 = Clock::now();
        std::cout << "1st version CPO Time : " << elapsed_1 / bench_iterations << " msec\n" << endl;
    }
    {
        clock_t t;
        t = clock();
        // auto t1 = Clock::now();
        for (int k = 0; k < bench_iterations; k++) CPO2(d_o3, filter_vectorized, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
        elapsed_2 = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds
        // auto t2 = Clock::now();
        std::cout << "2nd version CPO Time : " << elapsed_2 / bench_iterations << " msec\n" << endl;
    }
    // std::cout << "Diff Time for CPO : " << (elapsed_1 - elapsed_2) / bench_iterations << " milliseconds" << endl;
    {
        clock_t t;
        t = clock();
        // auto t1 = Clock::now();
        for (int k = 0; k < bench_iterations; k++) CPO3(d_o3, filter_vectorized, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
        // auto t2 = Clock::now();
        elapsed_3 = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds
        std::cout << "3rd version CPO Time : " << elapsed_3 / bench_iterations << " msec\n" << endl;
    }
    {
        clock_t t;
        t = clock();
        // auto t1 = Clock::now();
        for (int k = 0; k < bench_iterations; k++) _CPO(d_o3, filter_vectorized, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
        // auto t2 = Clock::now();
        elapsed_4 = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds
        std::cout << "Ahmed's CPO Time : " << elapsed_4 / bench_iterations << " msec\n" << endl;
    }

    {
        clock_t t;
        t = clock();
        // auto t1 = Clock::now();
        for (int k = 0; k < bench_iterations; k++) CSR1(d_o3, filter_vectorized, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
        // auto t2 = Clock::now();
        elapsed_5 = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds
        std::cout << "1st version CSR Time : " << elapsed_5 / bench_iterations << " msec\n" << endl;
    }

    {
        clock_t t;
        t = clock();
        // auto t1 = Clock::now();
        for (int k = 0; k < bench_iterations; k++) CSR1(d_o3, filter_vectorized, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
        // auto t2 = Clock::now();
        elapsed_6 = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds
        std::cout << "2nd version CSR Time : " << elapsed_6 / bench_iterations << " msec\n" << endl;
    }

    {
        clock_t t;
        t = clock();
        // auto t1 = Clock::now();
        for (int k = 0; k < bench_iterations; k++) Im2col(d_o1, filter_vectorized, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, Ic);
        elapsed_7 = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds
        // auto t2 = Clock::now();
        std::cout << "Im2col Time  : " << elapsed_7 / bench_iterations << " msec\n" << endl;
        // std::cout << "Im2col t2-t1: "
           //  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1000
           //  << " microsec\n" << std::endl;
    }
    std::cout << "Diff Time for CPO2 - CSR2 : " << (elapsed_3 - elapsed_6) / bench_iterations << " msec" << endl;
    std::cout << "Diff Time for CPO2 - Im2col : " << (elapsed_3 - elapsed_7) / bench_iterations << " msec" << endl;
    return 0;
}