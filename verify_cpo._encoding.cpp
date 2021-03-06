#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <boost/timer/timer.hpp>
#include <time.h>
#include <fstream>

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>


#define IS_PRINT        0
#define IS_PRINT_SIZE   0
#define case_1          1
using namespace Eigen;
using namespace std;
using namespace boost::timer;


void printVector(std::vector<int>& x)
{
    cout << "\nPrint 1D Vector" << endl;
    for (int i = 0; i < x.size(); ++i)
    {
        cout << x[i] << ", ";
    }
    cout << "\n";
}

// void transform2dTo1d_encoding(vector<vector<int> >& IN, vector<vector<int> >& DA, vector<int>& IN_1d, vector<int>& DA_1d)
void transform2dTo1d_encoding(vector<vector<int> >& IN, vector<vector<int> >& DA, vector<int>& IN_1d, vector<int>& DA_1d)
{
    // Convert the data and indices to 1-D:
    for (int i = 0; i < DA.size(); ++i)
    {
        const vector<int>& v1 = DA[i];
        DA_1d.insert(DA_1d.end(), v1.begin(), v1.end());

        const vector<int>& v2 = IN[i];
        IN_1d.insert(IN_1d.end(), v2.begin(), v2.end());

    }

}


// https://scicomp.stackexchange.com/questions/27977/how-can-i-speed-up-this-code-for-sparse-matrix-vector-multiplication

void bench_Sparse(const SparseMatrix<float>& m, const MatrixXf& in, MatrixXf& o) {
    // o.noalias() = m*in.transpose();
    o.noalias() = m * in;
}

void bench_Dense(const MatrixXf& m, const MatrixXf& in, MatrixXf& o) {
    // o.noalias() = m*in.transpose();

    // cout << m.rows() << ", " << m.cols() << endl;
    // cout << in.rows() << ", " << in.cols() << endl;
   ///  exit(0);
    o.noalias() = m * in;
    //  o.noalias() = m.transpose()*in;

}

void CSR(std::vector<int>& DA, std::vector<int>& IN, std::vector<int>& ptr, MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
    int i = 0;
    int x = 0; // 
    int m = 0; //submatrix

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
                m++;
                j = m - 1;
                ptr.push_back(x);
                if (j == Iw - Kw)
                {
                    break;
                }
            } // end if (i == Ih)
        } // end if (j == m + (Kw - 1))
    } // end for (int j = 0; j < Iw; ++j)

   // std::cout << "\nPtr: ";
   // printVector(ptr);

  //  std::cout << "\n\nIN: ";
  //  printVector(IN);

   // std::cout << "\n\nData:";
   // printVector(DA);
  //  std::cout << "\n" << endl;
}

void CPO5_OP(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw,
            vector<int>& ptr, vector<int>& IN_1d, vector<int>& DA_1d, vector<int>& x, vector<int>& m)
// (org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr_1d, x, m)
{
    // cout << " ############################### CPO 5 ###############################" << endl;
    // cout << "Initial Value for m" << endl;
    // printVector(m);
    int Ow_1 = (1 + Ow);
    int l_Ow_1;
    // int l = Kw-1; // Kw-1  Jump to the second part 
    int l = 1; // Kw-1  Jump to the second part 
    int npo_shift = Ow_1 - 3;  // as we igorned the repetition i
    int ind_val;
    // Initial value of the index of the pointer by 2 as all the pointer will reach the 
    // same index after the secind piece 
    int ptr_Indexhelper, x_helper, m_helper;
    // NPO
    {
        int x_helper = 0;
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, 0) != 0)
            {
                ind_val = (i * Kw); 
                IN_1d.push_back(ind_val);
                DA_1d.push_back(org_mat(i, 0));
                x_helper++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)       
        ptr[0] = 0;
        ptr[1] = x_helper;

        for (int i = 0; i < Ih; i++)
        {
            if (org_mat(i, Iw - 1) != 0)
            {
                ind_val = Kw - 1 + (i * Kw);
                IN_1d.push_back(ind_val);
                DA_1d.push_back(org_mat(i, Iw - 1));
                x_helper++;

            }// end if(org_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)
        ptr[2] = x_helper;
        // x[0] = x_helper; // no need as we have finished the NPO ... so no need to save it
    }

   
    // first piece & third piece 
    int x_helper1, x_helper2;
    // it should start from left to right ... so we need anther incremental for the last part 
    // of the image

    int j_ = Iw - 2;
    // cout << "2nd part merged" << endl;
    for (int j = 1 ; j < Kw-1; ++j)
    // for (int j = Kw-1 ; j <= 1; --j)
    {
        // First part
        x_helper1 = 0;
        // m_helper++;
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                ind_val = j + (i * Kw);
                IN_1d.push_back(ind_val);
                DA_1d.push_back(org_mat(i, j));


                x_helper1++;
            } // end if  if (org_mat(i, j) != 0)

        }
        
        // m_helper no need for it as all the pointers will be incremented by 1
        ptr_Indexhelper = j * Ow_1 - npo_shift + 1 ; 
        ptr[ptr_Indexhelper - 1] =  0;
        ptr[ptr_Indexhelper    ] =  x_helper1;
        x_helper2 = x_helper1;
        m_helper = m[l];
        for (int i = 0; i < Ih; ++i)
        {

            if (org_mat(i, j_) != 0)
            {                    
                ind_val = j_ - (m_helper - 1) + (i * Kw);
                IN_1d.push_back(ind_val);
                DA_1d.push_back(org_mat(i, j_));
                x_helper2++; // accum on the last value of the pointer 
            }// end if(org_mat(i, j) != 0)

        } // end for(i=0; i < Ih; ++i)       

        l_Ow_1 = l * Ow_1 - npo_shift;
        ptr[l_Ow_1 +  m_helper ] = x_helper2; 
        for (int c = 1; c < l; ++c)
        {
            m[c]--;
        } // end for(int c = 0; c < l-1; ++c)
        j_--;
        l++;
    } // end for (int j = 0; j < Kw; ++j)
    
    {
        // Second piece
        ptr_Indexhelper = l * Ow_1 - npo_shift;
        ptr[ptr_Indexhelper] = 0;
        m_helper = 1; // always the second element in the pointer 
        x_helper = 0;

        for (int j = Kw-1; j <= Iw - Kw; ++j)
        {
            for (int i = 0; i < Ih; ++i)
            {
                if (org_mat(i, j) != 0)
                {                    
                    ind_val = j - ( m_helper - 1) + (i * Kw);
                    IN_1d.push_back(ind_val);
                    DA_1d.push_back(org_mat(i, j));
                    x_helper++;
                } // end if (org_mat(i, j) != 0)
            }// end for(i = 0; i < Ih; ++i)
            // kernel size big ... pointer 
            // accessing is better at small size 
            ptr[ ptr_Indexhelper + m_helper] = x_helper; 
            m_helper++;
        } // end for(int j = Kw; j < Iw - Kw; ++j)
    }
} //CPO5

// CSR without eigen
void csrMult_v4(vector<vector<float> >& O, vector<int> const& K, vector<int>& Adata, vector<int>& Aindices, vector<int>& Aindptr, int Kh, int Kw, int Oh, int Ow)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    int x = Aindptr[0];
    int* Aindex_help = &Aindices[x];
    int* Adata_help = &Adata[x];
    for (int n = 0; n < Ow; ++n)
    {
        for (; x < Aindptr[n + 1]; ++x)
        {
            // double result = 0.0;
            int NZE_index = *Aindex_help; Aindex_help++;
            int NZE_data = *Adata_help; Adata_help++;


            for (int l = 0; l < Kh; ++l)
            {
                int m = NZE_index / Kw - l;
                int Kindex = NZE_index % Kw + l * Kw;
                if (m < 0 || m >= Oh) continue;

                // cout << "R) " << m << ", C) " << n << ", " << Kindex << endl;
                O[m][n] += NZE_data * K[Kindex];
            }
        }

    }
    // cout << "csrMult_v4 error" << endl;
} // end mult


void reset2DVectorF(std::vector<vector<float>>& x)
{
    for (int i = 0; i < x.size(); ++i)
    {
        for (int j = 0; j < x[i].size(); ++j)
        {
            x[i][j] = 0;
        }
    }
}

bool isEqualVectors(std::vector<vector<float>>& x1, std::vector<vector<float>>& x2)
{
    bool error = false;

    assert(x1.size() == x2.size());
    assert(x1[0].size() == x2[0].size());
    for (int i = 0; i < x1.size(); ++i)
    {
        for (int j = 0; j < x1[i].size(); ++j)
        {
            if (x1[i][j] != x2[i][j])
            {
                cout << "Error! " << i << ", " << j << endl;
                error = true;
                break;

            }
        }

        if (error)
            break;
    }

    return error;
}


void print2DVectorF(std::vector<vector<float>>& x)
{
    for (int i = 0; i < x.size(); ++i)
    {
        for (int j = 0; j < x[i].size(); ++j)
        {
            cout << x[i][j] << " ";
        }
        cout << "\n";
    }
}


void print2DVector(std::vector<vector<int>>& x)
{
    for (int i = 0; i < x.size(); ++i)
    {
        cout << "\n===== " << i << " =====\n";
        for (int j = 0; j < x[i].size(); ++j)
        {
            cout << x[i][j] << ", ";
        }
    }
}


/*SSSSSSSSSSSSSSSSSSSSSSSSSSS*/

void transform2dTo1d(vector<vector<int> >& IN, vector<vector<int> >& DA, vector<vector<int> >& ptr, vector<int>& IN_1d, vector<int>& DA_1d, vector<int>& ptr_1d)
{


    int c = 0;
    for (int i = 0; i < ptr.size(); ++i)
    {
        for (int j = 0; j < ptr[i].size(); ++j)
        {
            ptr_1d[c++] = ptr[i][j];
        }
    }

    int c1 = 0;
    for (int i = 0; i < DA.size(); ++i)
    {
        for (int j = 0; j < DA[i].size(); ++j)
        {
            DA_1d[c1] = DA[i][j];
            IN_1d[c1++] = IN[i][j];
        }
    }

}


void transform2dTo1dv1(vector<vector<int> >& IN, vector<vector<int> >& DA, vector<vector<int> >& ptr, vector<int>& IN_1d, vector<int>& DA_1d, vector<int>& ptr_1d)
{

    int c = 0;
    for (int i = 0; i < ptr.size(); ++i)
    {
        for (int j = 0; j < ptr[i].size(); ++j)
        {
            if (i == 0)
            {
                // Remove repeats
                if (j <= 1 || j == ptr[0].size() - 1)
                {
                    ptr_1d[c++] = ptr[i][j];
                }

            }
            else {
                ptr_1d[c++] = ptr[i][j];
            }

        }
    }

    int c1 = 0;
    for (int i = 0; i < DA.size(); ++i)
    {
        for (int j = 0; j < DA[i].size(); ++j)
        {
            DA_1d[c1] = DA[i][j];
            IN_1d[c1++] = IN[i][j];
        }
    }

}



void transform2dTo1dv9(vector<vector<int> >& IN, vector<vector<int> >& DA, vector<vector<int> >& ptr, vector<int>& IN_1d, vector<int>& DA_1d, vector<int>& ptr_1d)
{

    int c = 0;
    for (int i = 0; i < ptr.size(); ++i)
    {
        for (int j = 0; j < ptr[i].size(); ++j)
        {
            // if(i == 0)
            // For all ptrs except the last one
            if (i <= ptr.size() - 2)
            {
                // Remove repeats
                if (j <= 1 || j == ptr[i].size() - 1)
                {
                    ptr_1d[c++] = ptr[i][j];
                }

            }
            else {
                ptr_1d[c++] = ptr[i][j];
            }

        }
    }

    int c1 = 0;
    for (int i = 0; i < DA.size(); ++i)
    {
        for (int j = 0; j < DA[i].size(); ++j)
        {
            DA_1d[c1] = DA[i][j];
            IN_1d[c1++] = IN[i][j];
        }
    }

}


void conv_CPO_v9_trimV2(vector<vector<float> >& O, vector<int> const& K, vector<int>& IN, vector<int>& DA, vector<int>& ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{

    const int n = ceil(Kw / Sw); // n is the number of ptr (NPO, PO2, PO3)
    const int number = floor((Iw - Kw) / Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;

    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int* x_ptr = &ptr[0];
    int  x = *x_ptr;
    int* Aindex_help = &IN[x];
    int* Adata_help = &DA[x];

    // For each ptr type
    // for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    int type_ptr = 0;
    {

        // Submat 0
        x = *x_ptr;
        int end_x_loop = *(x_ptr + 1);
        ++x_ptr;

        // x loop
       // l loop
             // cout << "V9) Sumbat: " << 0 << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
        for (; x < end_x_loop; ++x)
        {


            // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
             // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

             // How many time to iterate?
            int used_index = *Aindex_help; Aindex_help++;
            int used_data = *Adata_help;  Adata_help++;

            // int shereet2 = min(submat, type_ptr); 
            for (int l = 0; l < Kh; ++l)
            {
                // I = 0:
               // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                int y_out = (used_index / Kw) - l;

                // 2, 1, 0
                // cout << "Y_out " << y_out << endl;
                if (y_out >= 0 && y_out < Oh) {
                    // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
                    O[y_out][0] += used_data * K[used_index % Kw + l * Kw];
                } // end if(y_out >= 0 && y_out < Oh) 

            } // end l loop

        } // end x

        // Submat number - 1
        x = *x_ptr;
        end_x_loop = *(x_ptr + 1);
        ++x_ptr;


        // cout << "V9) Sumbat: " << (number-1) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
        // x loop
        // l loop
              // cout << "V7) Sumbat: " << (number) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
        for (; x < end_x_loop; ++x)
        {


            // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
             // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

             // How many time to iterate?
            int used_index = *Aindex_help; Aindex_help++;
            int used_data = *Adata_help;  Adata_help++;

            // int shereet2 = min(submat, type_ptr); 
            for (int l = 0; l < Kh; ++l)
            {
                // I = 0:
               // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                int y_out = (used_index / Kw) - l;

                // 2, 1, 0
                // cout << "Y_out " << y_out << endl;
                if (y_out >= 0 && y_out < Oh) {
                    // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
                    O[y_out][number - 1] += used_data * K[used_index % Kw + l * Kw];
                } // end if(y_out >= 0 && y_out < Oh) 

            } // end l loop

        } // end x


        ++x_ptr;
    }

    // For each ptr type
    // int type_ptr = 0;
    for (int type_ptr = 1; type_ptr < n - 1; ++type_ptr)
    {


        // Submat 0:
        x = *x_ptr;
        int end_x_loop = *(x_ptr + 1);
        ++x_ptr;

        // x loop
       // l loop
       // cout << "V7) Sumbat: " << 0 << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    

       // cout << "V9) Sumbat: " << (0) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
        for (; x < end_x_loop; ++x)
        {


            // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
             // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

             // How many time to iterate?
            int used_index = *Aindex_help; Aindex_help++;
            int used_data = *Adata_help;  Adata_help++;

            for (int l = 0; l < Kh; ++l)
            {
                // I = 0:
               // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                int y_out = (used_index / Kw) - l;

                // 2, 1, 0
                // cout << "Y_out " << y_out << endl;
                if (y_out >= 0 && y_out < Oh)
                {
                    // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
                    O[y_out][0] += used_data * K[used_index % Kw + l * Kw];


                    // for loop for i and sumbat 0
                    int kernel_common_index = used_index % Kw + l * Kw;

                    for (int i = 1; i <= type_ptr; ++i)
                    {
                        O[y_out][0 + i] += used_data * K[kernel_common_index - i];

                    } // end i for loop

                } // end if(y_out >= 0 && y_out < Oh) 

            } // end l loop
        } // end x


        // Submat number - 1:
        x = *x_ptr;
        end_x_loop = *(x_ptr + 1);
        ++x_ptr;

        // cout << "V9) Sumbat: " << (number-1) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    

        for (; x < end_x_loop; ++x)
        {


            // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
             // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

             // How many time to iterate?
            int used_index = *Aindex_help; Aindex_help++;
            int used_data = *Adata_help;  Adata_help++;

            for (int l = 0; l < Kh; ++l)
            {
                // I = 0:
               // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                int y_out = (used_index / Kw) - l;

                // 2, 1, 0
                // cout << "Y_out " << y_out << endl;
                if (y_out >= 0 && y_out < Oh)
                {
                    // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
                    O[y_out][number - 1 - type_ptr] += used_data * K[used_index % Kw + l * Kw];


                    // for loop for i and sumbat 0
                    int kernel_common_index = used_index % Kw + l * Kw;
                    int out_common_index = number - 1 - type_ptr;

                    for (int i = 1; i <= type_ptr; ++i)
                    {
                        O[y_out][out_common_index + i] += used_data * K[kernel_common_index - i];

                    } // end i for loop

                } // end if(y_out >= 0 && y_out < Oh) 

            } // end l loop
        } // end x


        ++x_ptr;
    } // end type ptr first loop


    // Last Type ptr
    // for (int type_ptr = 10000; type_ptr < n; ++type_ptr)
    type_ptr = n - 1;
    {


        // For each submat
        for (int submat = 0; submat < number; ++submat)
        {

            x = *x_ptr;
            int end_x_loop = *(x_ptr + 1);
            ++x_ptr;

            // cout << "V9) Sumbat: " << submat << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;   
            // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;


            for (; x < end_x_loop; ++x)
            {


                // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
                 // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

                 // How many time to iterate?
                int used_index = *Aindex_help; Aindex_help++;
                int used_data = *Adata_help;  Adata_help++;

                // int shereet2 = min(submat, type_ptr); 
                for (int l = 0; l < Kh; ++l)
                {
                    // I = 0:
                   // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                    int input_index = used_index;
                    int y_out = (input_index / Kw) - l;
                    int x_out = submat;
                    int kernel_common_index = input_index % Kw + l * Kw;

                    if (y_out >= 0 && y_out < Oh) {
                        O[y_out][x_out] += used_data * K[input_index % Kw + l * Kw];


                        for (int i = 1; i <= type_ptr; ++i)
                        {


                            // O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 
                            // O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 

                            O[y_out][x_out + i] += used_data * K[kernel_common_index - i];

                            // int input_index     = used_index - i;
                            // O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 

                            // cout << "l: " << l  << " i: " << i <<  " KERNEL DEBUG: " << " input_index: " << input_index << " Kw: " << Kw <<  " l*Kw: " 
                            // << l*Kw <<  " (input_index mod Kw): " << (input_index%Kw) << " first i: " << (used_index%Kw + l*Kw) << " current i: " << (input_index%Kw + l*Kw) << endl;

                            // i = 2;
                            // input_index     = used_index - i;
                            // O[y_out][x_out + 2] += used_data * K[input_index%Kw + l*Kw]; 


                        } // end i for loop

                    } // end if(y_out >= 0 && y_out < Oh) 
                    // cout << "\n";

                } // end l loop

            } // end x
        } // end sumbat

        ++x_ptr;
    } // end type ptr
} // end conv_CPO_v9

int CPO4(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> >& IN,
    vector<vector<int> >& DA, vector<int>& ptr, vector<int>& x, vector<int>& m)
{
    // cout << "Start" << endl;
    int Ow_1 = (1 + Ow);
    // int l = 1; // Kw-1
    int l = Kw - 1; // Kw-1
    int n = Kw;
    int non_zero_count = 0;
    int npo_shift = Ow_1 - 3;

    int ind_val;
    // Initial value of the index of the pointer by 2 as all the pointer will reach the 
    // same index after the secind piece 

    int ptr_Indexhelper, x_helper, m_helper;
    // NPO
    {
        // int j = 0;
        int x_helper = 0;
        // m[l]++;
        // (*m_ptr)++;
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, 0) != 0)
            {
                ind_val = (i * Kw);
                IN[0].push_back(ind_val);
                DA[0].push_back(org_mat(i, 0));
                // x[l]++;
                x_helper++;
                non_zero_count++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)       
        // ptr2d[l][m[l]] = x[l];
        // ptr[l * (1 + Ow) + m[l]] = x[l];
        ptr[0] = 0;
        ptr[1] = x_helper;

        for (int i = 0; i < Ih; i++)
        {
            if (org_mat(i, Iw - 1) != 0)
            {
                ind_val = Kw - 1 + (i * Kw);
                IN[0].push_back(ind_val);
                DA[0].push_back(org_mat(i, Iw - 1));
                x_helper++;
                non_zero_count++;

            }// end if(org_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)
        ptr[2] = x_helper;
        x[0] = x_helper;
    }

    // first piece
    int x_helper1;

    // cout << "help !!" << endl;
    // printVector(x);
    // printVector(m);
    // ptr_Indexhelper = l * Ow_1 - npo_shift + 1;

    for (int j = 1; j < Kw; ++j)
    {
        x_helper1 = 0;
        // m_helper = m[l];
        // m_helper++;
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                ind_val = j + (i * Kw);
                IN[j].push_back(ind_val);
                DA[j].push_back(org_mat(i, j));
                // x[l]++;
                // (*x_ptr)++;
                x_helper1++;
                non_zero_count++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)       
        // ptr_Indexhelper = l * Ow_1 - npo_shift + m_helper;

        // m_helper no need for it as all the pointers will be incremented by 1
        // ptr_Indexhelper = l * (1 + Ow) - npo_shift + m_helper ( 1 );
        ptr_Indexhelper = j * Ow_1 - npo_shift + 1;
        ptr[ptr_Indexhelper - 1] = 0;
        ptr[ptr_Indexhelper] = x_helper1;
        x[j] = x_helper1;
        // m_helper++;
        // m[l] = m_helper + 1;
        // l++;
    } // end for (int j = 0; j < Kw; ++j)

    // l--;

    {
        // Second piece
        ptr_Indexhelper = l * Ow_1 - npo_shift;
        // m_helper = m[l];
        m_helper = 2;
        x_helper = x[l];
        // int *ptr_Index = &ptr[ ptr_Indexhelper + m_helper];
        // int *ptr_Index = &ptr[ ptr_Indexhelper + 2];
        for (int j = Kw; j < Iw - Kw; ++j)
        {
            for (int i = 0; i < Ih; ++i)
            {
                if (org_mat(i, j) != 0)
                {
                    // ind_val = j - ( (*m_ptr) - 1) + (i * Kw);
                    ind_val = j - (m_helper - 1) + (i * Kw);
                    IN[l].push_back(ind_val);
                    DA[l].push_back(org_mat(i, j));
                    // x[l]++;
                    // (*x_ptr)++;
                    x_helper++;
                    non_zero_count++;
                } // end if (org_mat(i, j) != 0)
            }// end for(i = 0; i < Ih; ++i)

            // kernel size big ... pointer 
            // accessing is better at small size 
            ptr[ptr_Indexhelper + m_helper] = x_helper;
            // *(ptr_Index++)=x_helper;
            m_helper++;


        } // end for(int j = Kw; j < Iw - Kw; ++j)
        x[l] = x_helper;
        m[l] = m_helper;
    }


    // cout << "\nafter stage 2 !!!!" << endl;
    // printVector(m);


    // increment the position of col in the pointer of PO2
    // repeating elements for all pointers except NPO//last pointer
    // for (int c = 1; c < n - 1; ++c)
    // {
    //     // ptr2d[c][m[c]] = x[c];
    //     // ptr[c * (1 + Ow) + m[c] - npo_shift] = x[c];
    //     m[c] += Iw - 2*Kw + 1;
    //     // m[c] += Iw - 2*Kw ;
    // } // end for(int c = 0; c < n-1; ++c)

    int l_Ow_1;
    // cout << "before the 3rd piece : " << endl;
    // printVector(m);
    // std::vector<int>::iterator iter_begin = m.begin() + 1;
    // Third piece   
    for (int j = Iw - Kw; j < Iw - 1; ++j)
    {
        x_helper1 = x[l];
        // x_helper = x_helper1;
        m_helper = m[l];
        // cout << l <<" Img col : " << j << endl;
        for (int i = 0; i < Ih; i++)
        {
            if (org_mat(i, j) != 0)
            {
                ind_val = j - (m_helper - 1) + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x_helper1++; // accum on the last value of the pointer 
                non_zero_count++;

            }// end if(org_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)

            // cout << "ptr index : " << l * Ow_1 + m_helper - npo_shift << endl;

        // ptr[l * Ow_1 + m_helper - npo_shift] = x_helper;
        // m[l] = m_helper + l;
        // // cout << " m[l] = m_helper + l; " << endl;
        // // printVector(m);
        // x[l] = x_helper;
        l_Ow_1 = l * Ow_1 - npo_shift;
        ptr[l_Ow_1 + m_helper] = x_helper1;
        // ptr[l_Ow_1 +  m_helper - 1 ] = x_helper;
        // for (int c = 0; c < l + 1; ++c) //<----
        // {
        //     // ptr[l][m[l]] = x[l];
        //     ptr[l_Ow_1 +  m_helper ] = x_helper; //<---
        //     m_helper++;
        // } // end for(int c = 0; c < l+1; ++c)

        // std::vector<int>::iterator iter_end = std::next(m.begin(), l);
        // std::transform(iter_begin, iter_end , iter_begin, bind2nd(std::plus<int>(), 1)); 
        for (int c = 1; c < l; ++c)
        {
            // ptr[c][m[c]] = x[c];
            // ptr[c*Ow_1 +  m[c] - npo_shift] = x[c]; //<---
            m[c]++;
        } // end for(int c = 0; c < l-1; ++c)

        l--;

        // cout << "\nThird p : "<< j << endl;
        // printVector(m);
    } // end for (j = Iw - Kw; Iw; ++j)

#if 0
   // std::cout << "\nPtr: ";
  // print2DVector(ptr);

   // std::cout << "\n\nIN: ";
  // print2DVector(IN);

   //  std::cout << "\n\nData:";
    //  print2DVector(DA);
   // std::cout << "\n" << endl;
#endif 
    return non_zero_count;

} //CPO4 


void conv_CPO_v9_trim(vector<vector<float> >& O, vector<int> const& K, vector<int>& IN, vector<int>& DA, vector<int>& ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    const int n = ceil(Kw / Sw); // n is the number of ptr (NPO, PO2, PO3)
    const int number = floor((Iw - Kw) / Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;

    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int* x_ptr = &ptr[0];
    int  x = *x_ptr;
    int* Aindex_help = &IN[x];
    int* Adata_help = &DA[x];

    // For each ptr type
    // int type_ptr = 0;
    for (int type_ptr = 0; type_ptr < n - 1; ++type_ptr)
    {


        // Submat 0:
        x = *x_ptr;
        int end_x_loop = *(x_ptr + 1);
        ++x_ptr;

        // x loop
       // l loop
       // cout << "V7) Sumbat: " << 0 << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    

       // cout << "V9) Sumbat: " << (0) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
        for (; x < end_x_loop; ++x)
        {


            // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
             // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

             // How many time to iterate?
            int used_index = *Aindex_help; Aindex_help++;
            int used_data = *Adata_help;  Adata_help++;

            for (int l = 0; l < Kh; ++l)
            {
                // I = 0:
               // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                int y_out = (used_index / Kw) - l;

                // 2, 1, 0
                // cout << "Y_out " << y_out << endl;
                if (y_out >= 0 && y_out < Oh)
                {
                    // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
                    O[y_out][0] += used_data * K[used_index % Kw + l * Kw];


                    // for loop for i and sumbat 0
                    int kernel_common_index = used_index % Kw + l * Kw;

                    for (int i = 1; i <= type_ptr; ++i)
                    {
                        O[y_out][i] += used_data * K[kernel_common_index - i];

                    } // end i for loop

                } // end if(y_out >= 0 && y_out < Oh) 

            } // end l loop
        } // end x


        // Submat number - 1:
        x = *x_ptr;
        end_x_loop = *(x_ptr + 1);
        ++x_ptr;

        // cout << "V9) Sumbat: " << (number-1) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    

        for (; x < end_x_loop; ++x)
        {


            // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
             // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

             // How many time to iterate?
            int used_index = *Aindex_help; Aindex_help++;
            int used_data = *Adata_help;  Adata_help++;

            for (int l = 0; l < Kh; ++l)
            {
                // I = 0:
               // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                int y_out = (used_index / Kw) - l;

                // 2, 1, 0
                // cout << "Y_out " << y_out << endl;
                if (y_out >= 0 && y_out < Oh)
                {
                    // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
                    O[y_out][number - 1 - type_ptr] += used_data * K[used_index % Kw + l * Kw];


                    // for loop for i and sumbat 0
                    int kernel_common_index = used_index % Kw + l * Kw;

                    for (int i = 1; i <= type_ptr; ++i)
                    {
                        O[y_out][number - 1 - type_ptr + i] += used_data * K[kernel_common_index - i];

                    } // end i for loop

                } // end if(y_out >= 0 && y_out < Oh) 

            } // end l loop
        } // end x


        ++x_ptr;
    } // end type ptr first loop


    // Last Type ptr
    // for (int type_ptr = 10000; type_ptr < n; ++type_ptr)
    int type_ptr = n - 1;
    {


        // For each submat
        for (int submat = 0; submat < number; ++submat)
        {

            x = *x_ptr;
            int end_x_loop = *(x_ptr + 1);
            ++x_ptr;

            // cout << "V9) Sumbat: " << submat << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;   
            // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;


            for (; x < end_x_loop; ++x)
            {


                // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
                 // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

                 // How many time to iterate?
                int used_index = *Aindex_help; Aindex_help++;
                int used_data = *Adata_help;  Adata_help++;

                // int shereet2 = min(submat, type_ptr); 
                for (int l = 0; l < Kh; ++l)
                {
                    // I = 0:
                   // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                    int input_index = used_index;
                    int y_out = (input_index / Kw) - l;
                    int x_out = submat;
                    int kernel_common_index = input_index % Kw + l * Kw;

                    if (y_out >= 0 && y_out < Oh) {
                        O[y_out][x_out] += used_data * K[input_index % Kw + l * Kw];


                        for (int i = 1; i <= type_ptr; ++i)
                        {


                            // O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 
                            // O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 

                            O[y_out][x_out + i] += used_data * K[kernel_common_index - i];

                            // int input_index     = used_index - i;
                            // O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 

                            // cout << "l: " << l  << " i: " << i <<  " KERNEL DEBUG: " << " input_index: " << input_index << " Kw: " << Kw <<  " l*Kw: " 
                            // << l*Kw <<  " (input_index mod Kw): " << (input_index%Kw) << " first i: " << (used_index%Kw + l*Kw) << " current i: " << (input_index%Kw + l*Kw) << endl;

                            // i = 2;
                            // input_index     = used_index - i;
                            // O[y_out][x_out + 2] += used_data * K[input_index%Kw + l*Kw]; 


                        } // end i for loop

                    } // end if(y_out >= 0 && y_out < Oh) 
                    // cout << "\n";

                } // end l loop

            } // end x
        } // end sumbat

        ++x_ptr;
    } // end type ptr


} // end conv_CPO_v9

void conv_CPO_v8_trim(vector<vector<float> >& O, vector<int> const& K, vector<int>& IN, vector<int>& DA, vector<int>& ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    const int n = ceil(Kw / Sw); // n is the number of ptr (NPO, PO2, PO3)
    const int number = floor((Iw - Kw) / Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;

    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int* x_ptr = &ptr[0];
    int last_prev;
    int  x = *x_ptr;
    int* Aindex_help = &IN[x];
    int* Adata_help = &DA[x];

    // For each ptr type
    // for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    int type_ptr = 0;
    {

        // Submat 0
         // if -1
        x = *x_ptr;
        int end_x_loop = *(x_ptr + 1);
        ++x_ptr;

        // x loop
       // l loop
       // cout << "V7) Sumbat: " << 0 << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
        for (; x < end_x_loop; ++x)
        {


            // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  

             // cout << "\n" << type_ptr << " ==> Current Submat " << 0 << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + 0 + 1 << endl;  

             // How many time to iterate?
            int used_index = *Aindex_help; Aindex_help++;
            int used_data = *Adata_help;  Adata_help++;

            // int shereet2 = min(submat, type_ptr); 
            for (int l = 0; l < Kh; ++l)
            {
                // I = 0:
               // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                int y_out = (used_index / Kw) - l;

                // 2, 1, 0
                // cout << "Y_out " << y_out << endl;
                if (y_out >= 0 && y_out < Oh) {
                    // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
                    O[y_out][0] += used_data * K[used_index % Kw + l * Kw];
                } // end if(y_out >= 0 && y_out < Oh) 

            } // end l loop

        } // end x

        // Submat number - 1

        // if -1
        x = *x_ptr;
        end_x_loop = *(x_ptr + 1);
        ++x_ptr;


        // cout << "V7) Sumbat: " << (number-1) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
        // x loop
        // l loop
              // cout << "V7) Sumbat: " << (number) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
        for (; x < end_x_loop; ++x)
        {


            // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
             // cout << "\n" << type_ptr << " ==> Current Submat " << (number-1) << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + (number-1)  + 1 << endl;  

             // How many time to iterate?
            int used_index = *Aindex_help; Aindex_help++;
            int used_data = *Adata_help;  Adata_help++;

            // int shereet2 = min(submat, type_ptr); 
            for (int l = 0; l < Kh; ++l)
            {
                // I = 0:
               // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                int y_out = (used_index / Kw) - l;

                // 2, 1, 0
                // cout << "Y_out " << y_out << endl;
                if (y_out >= 0 && y_out < Oh) {
                    // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
                    O[y_out][number - 1] += used_data * K[used_index % Kw + l * Kw];
                } // end if(y_out >= 0 && y_out < Oh) 

            } // end l loop

        } // end x


        ++x_ptr;
    }
    // cout << "conv_CPO_v8_trim ERROR" << endl; 
    // Type ptr 1
    // for (int type_ptr = 10000; type_ptr < n; ++type_ptr)
    for (int type_ptr = 1; type_ptr < n; ++type_ptr)
    {


        // For each submat
        for (int submat = 0; submat < number; ++submat)
        {
            // if -1

            x = *x_ptr;
            int end_x_loop = *(x_ptr + 1);
            if (end_x_loop < 0) { *(x_ptr + 1) = x; }

            ++x_ptr;
            // if (x<0) continue; // After the ++x_ptr;


            // cout << "V7) Sumbat: " << submat << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;   
            // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;


            for (; x < end_x_loop; ++x)
            {


                // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
                 // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

                 // How many time to iterate?
                int used_index = *Aindex_help; Aindex_help++;
                int used_data = *Adata_help;  Adata_help++;

                // int shereet2 = min(submat, type_ptr); 
                for (int l = 0; l < Kh; ++l)
                {
                    // I = 0:
                   // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                    int input_index = used_index;
                    int y_out = (input_index / Kw) - l;
                    int x_out = submat;
                    int kernel_common_index = input_index % Kw + l * Kw;

                    if (y_out >= 0 && y_out < Oh) {
                        O[y_out][x_out] += used_data * K[input_index % Kw + l * Kw];


                        for (int i = 1; i <= type_ptr; ++i)
                        {


                            int input_index = used_index - i;
                            // O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 
                            // O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 
                            O[y_out][x_out + i] += used_data * K[kernel_common_index - i];

                            // cout << "l: " << l  << " i: " << i <<  " KERNEL DEBUG: " << " input_index: " << input_index << " Kw: " << Kw <<  " l*Kw: " 
                            // << l*Kw <<  " (input_index mod Kw): " << (input_index%Kw) << " first i: " << (used_index%Kw + l*Kw) << " current i: " << (input_index%Kw + l*Kw) << endl;

                            // i = 2;
                            // input_index     = used_index - i;
                            // O[y_out][x_out + 2] += used_data * K[input_index%Kw + l*Kw]; 


                        } // end i for loop

                    } // end if(y_out >= 0 && y_out < Oh) 
                    // cout << "\n";

                } // end l loop

            } // end x
        } // end sumbat

        ++x_ptr;
    } // end type ptr

      // cout << "Output: " << endl;
      // print2DVectorF(O);
      // cout << "-----\n" << endl;
}

void conv_CPO_v8_trim_(vector<vector<float> >& O, vector<int> const& K, vector<int>& IN, vector<int>& DA, vector<int>& ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    const int n = ceil(Kw / Sw); // n is the number of ptr (NPO, PO2, PO3)
    const int number = floor((Iw - Kw) / Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;

    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int* x_ptr = &ptr[0];
    int last_prev;
    int  x = *x_ptr;
    int* Aindex_help = &IN[x];
    int* Adata_help = &DA[x];

    // For each ptr type
    // for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    int type_ptr = 0;
    {

        // Submat 0
         // if -1
        x = *x_ptr;
        int end_x_loop = *(x_ptr + 1);
        ++x_ptr;

        // x loop
       // l loop
        cout << "V7) Sumbat: " << 0 << ", type_ptr: " << type_ptr << " start " << x << " end: " << end_x_loop << endl;
        for (; x < end_x_loop; ++x)
        {


            // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  

            cout << "\n" << type_ptr << " ==> Current Submat " << 0 << ", ptr:  " << x << ", ptr+1: " << end_x_loop << " ind to fetch: " << type_ptr * number + 0 + 1 << endl;

            // How many time to iterate?
            int used_index = *Aindex_help; Aindex_help++;
            int used_data = *Adata_help;  Adata_help++;

            // int shereet2 = min(submat, type_ptr); 
            for (int l = 0; l < Kh; ++l)
            {
                // I = 0:
               // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                int y_out = (used_index / Kw) - l;

                // 2, 1, 0
                // cout << "Y_out " << y_out << endl;
                if (y_out >= 0 && y_out < Oh) {
                    // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
                    O[y_out][0] += used_data * K[used_index % Kw + l * Kw];
                } // end if(y_out >= 0 && y_out < Oh) 

            } // end l loop

        } // end x

        // Submat number - 1

        // if -1
        x = *x_ptr;
        end_x_loop = *(x_ptr + 1);
        ++x_ptr;


        cout << "V7) Sumbat: " << (number - 1) << ", type_ptr: " << type_ptr << " start " << x << " end: " << end_x_loop << endl;
        // x loop
        // l loop
              // cout << "V7) Sumbat: " << (number) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
        for (; x < end_x_loop; ++x)
        {


            // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
            cout << "\n" << type_ptr << " ==> Current Submat " << (number - 1) << ", ptr:  " << x << ", ptr+1: " << end_x_loop << " ind to fetch: " << type_ptr * number + (number - 1) + 1 << endl;

            // How many time to iterate?
            int used_index = *Aindex_help; Aindex_help++;
            int used_data = *Adata_help;  Adata_help++;

            // int shereet2 = min(submat, type_ptr); 
            for (int l = 0; l < Kh; ++l)
            {
                // I = 0:
               // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                int y_out = (used_index / Kw) - l;

                // 2, 1, 0
                // cout << "Y_out " << y_out << endl;
                if (y_out >= 0 && y_out < Oh) {
                    // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
                    O[y_out][number - 1] += used_data * K[used_index % Kw + l * Kw];
                } // end if(y_out >= 0 && y_out < Oh) 

            } // end l loop

        } // end x


        ++x_ptr;
    }

    // Type ptr 1
    // for (int type_ptr = 10000; type_ptr < n; ++type_ptr)
    for (int type_ptr = 1; type_ptr < n; ++type_ptr)
    {


        // For each submat
        for (int submat = 0; submat < number; ++submat)
        {
            // if -1

            x = *x_ptr;
            int end_x_loop = *(x_ptr + 1);

            if (x < 0)
            {
                x = last_prev;
            }

            if (end_x_loop < 0)
            {
                last_prev = x;
            }

            ++x_ptr;

            cout << "V7) Sumbat: " << submat << ", type_ptr: " << type_ptr << " start " << x << " end: " << end_x_loop << endl;
            // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;


            for (; x < end_x_loop; ++x)
            {


                // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
                cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " << x << ", ptr+1: " << end_x_loop << " ind to fetch: " << type_ptr * number + submat + 1 << endl;

                // How many time to iterate?
                int used_index = *Aindex_help; Aindex_help++;
                int used_data = *Adata_help;  Adata_help++;

                // int shereet2 = min(submat, type_ptr); 
                for (int l = 0; l < Kh; ++l)
                {
                    // I = 0:
                   // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                    int input_index = used_index;
                    int y_out = (input_index / Kw) - l;
                    int x_out = submat;
                    int kernel_common_index = input_index % Kw + l * Kw;

                    if (y_out >= 0 && y_out < Oh) {
                        O[y_out][x_out] += used_data * K[input_index % Kw + l * Kw];


                        for (int i = 1; i <= type_ptr; ++i)
                        {


                            int input_index = used_index - i;
                            // O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 
                            // O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 
                            O[y_out][x_out + i] += used_data * K[kernel_common_index - i];

                            // cout << "l: " << l  << " i: " << i <<  " KERNEL DEBUG: " << " input_index: " << input_index << " Kw: " << Kw <<  " l*Kw: " 
                            // << l*Kw <<  " (input_index mod Kw): " << (input_index%Kw) << " first i: " << (used_index%Kw + l*Kw) << " current i: " << (input_index%Kw + l*Kw) << endl;

                            // i = 2;
                            // input_index     = used_index - i;
                            // O[y_out][x_out + 2] += used_data * K[input_index%Kw + l*Kw]; 


                        } // end i for loop

                    } // end if(y_out >= 0 && y_out < Oh) 
                    // cout << "\n";

                } // end l loop

            } // end x
        } // end sumbat

        ++x_ptr;
    } // end type ptr

      // cout << "Output: " << endl;
      // print2DVectorF(O);
      // cout << "-----\n" << endl;
}

// Removing repeats + previous optimizations
void conv_CPO_v7(vector<vector<float> >& O, vector<int> const& K, vector<int>& IN, vector<int>& DA, vector<int>& ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    // printVector(ptr);
    // exit(0);

    int n = ceil(Kw / Sw); // n is the number of ptr (NPO, PO2, PO3)
    int number = floor((Iw - Kw) / Sw) + 1; // number of elements in each ptr

    // number = 0; n = 1;

    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int* x_ptr = &ptr[0];
    int  x = *x_ptr;
    int* Aindex_help = &IN[x];
    int* Adata_help = &DA[x];

    // For each ptr type
    // for (int type_ptr = 0; type_ptr < n; ++type_ptr)

    // For each submat
    for (int submat = 0; submat < number; ++submat)
    {

        x = *x_ptr;
        int end_x_loop = *(x_ptr + 1);
        ++x_ptr;

        // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;


        for (; x < end_x_loop; ++x)
        {


            // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
             // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

             // How many time to iterate?
            int used_index = *Aindex_help; Aindex_help++;
            int used_data = *Adata_help;  Adata_help++;

            // int shereet2 = min(submat, type_ptr); 
            for (int l = 0; l < Kh; ++l)
            {
                // I = 0:
               // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                int input_index = used_index;
                int y_out = (input_index / Kw) - l;
                int x_out = submat;

                if (y_out >= 0 && y_out < Oh) {
                    O[y_out][x_out] += used_data * K[input_index % Kw + l * Kw];
                } // end if(y_out >= 0 && y_out < Oh) 

            } // end l loop

        } // end x
    } // end sumbat


    // cout << "Output: " << endl;
    // print2DVectorF(O);
    // cout << "-----\n" << endl;
}



void conv_CPO_v7_trim(vector<vector<float> >& O, vector<int> const& K, vector<int>& IN, vector<int>& DA, vector<int>& ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    const int n = ceil(Kw / Sw); // n is the number of ptr (NPO, PO2, PO3)
    const int number = floor((Iw - Kw) / Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;

    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int* x_ptr = &ptr[0];
    int  x = *x_ptr;
    int* Aindex_help = &IN[x];
    int* Adata_help = &DA[x];

    // For each ptr type
    // for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    int type_ptr = 0;
    {

        // Submat 0
        x = *x_ptr;
        int end_x_loop = *(x_ptr + 1);
        ++x_ptr;

        // x loop
       // l loop
             // cout << "V7) Sumbat: " << 0 << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
        for (; x < end_x_loop; ++x)
        {


            // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
             // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

             // How many time to iterate?
            int used_index = *Aindex_help; Aindex_help++;
            int used_data = *Adata_help;  Adata_help++;

            // int shereet2 = min(submat, type_ptr); 
            for (int l = 0; l < Kh; ++l)
            {
                // I = 0:
               // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                int y_out = (used_index / Kw) - l;

                // 2, 1, 0
                // cout << "Y_out " << y_out << endl;
                if (y_out >= 0 && y_out < Oh) {
                    // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
                    O[y_out][0] += used_data * K[used_index % Kw + l * Kw];
                } // end if(y_out >= 0 && y_out < Oh) 

            } // end l loop

        } // end x

        // Submat number - 1
        x = *x_ptr;
        end_x_loop = *(x_ptr + 1);
        ++x_ptr;


        // cout << "V7) Sumbat: " << (number-1) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
        // x loop
        // l loop
              // cout << "V7) Sumbat: " << (number) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
        for (; x < end_x_loop; ++x)
        {


            // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
             // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

             // How many time to iterate?
            int used_index = *Aindex_help; Aindex_help++;
            int used_data = *Adata_help;  Adata_help++;

            // int shereet2 = min(submat, type_ptr); 
            for (int l = 0; l < Kh; ++l)
            {
                // I = 0:
               // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                int y_out = (used_index / Kw) - l;

                // 2, 1, 0
                // cout << "Y_out " << y_out << endl;
                if (y_out >= 0 && y_out < Oh) {
                    // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
                    O[y_out][number - 1] += used_data * K[used_index % Kw + l * Kw];
                } // end if(y_out >= 0 && y_out < Oh) 

            } // end l loop

        } // end x


        ++x_ptr;
    }

    // Type ptr 1
    // for (int type_ptr = 10000; type_ptr < n; ++type_ptr)
    for (int type_ptr = 1; type_ptr < n; ++type_ptr)
    {


        // For each submat
        for (int submat = 0; submat < number; ++submat)
        {

            x = *x_ptr;
            int end_x_loop = *(x_ptr + 1);
            ++x_ptr;

            // cout << "V7) Sumbat: " << submat << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;   
            // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;


            for (; x < end_x_loop; ++x)
            {


                // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
                 // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

                 // How many time to iterate?
                int used_index = *Aindex_help; Aindex_help++;
                int used_data = *Adata_help;  Adata_help++;

                // int shereet2 = min(submat, type_ptr); 
                for (int l = 0; l < Kh; ++l)
                {
                    // I = 0:
                   // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                    int input_index = used_index;
                    int y_out = (input_index / Kw) - l;
                    int x_out = submat;


                    if (y_out >= 0 && y_out < Oh) {
                        O[y_out][x_out] += used_data * K[input_index % Kw + l * Kw];

                        for (int i = 1; i <= type_ptr; ++i)
                        {
                            int input_index = used_index - i;
                            O[y_out][x_out + i] += used_data * K[input_index % Kw + l * Kw];

                            // i = 2;
                            // input_index     = used_index - i;
                            // O[y_out][x_out + 2] += used_data * K[input_index%Kw + l*Kw]; 


                        } // end i for loop
                    } // end if(y_out >= 0 && y_out < Oh) 

                } // end l loop

            } // end x
        } // end sumbat

        ++x_ptr;
    } // end type ptr

      // cout << "Output: " << endl;
      // print2DVectorF(O);
      // cout << "-----\n" << endl;
}



void conv_CPO_v6(vector<vector<float> >& O, vector<int> const& K, vector<int>& IN, vector<int>& DA, vector<int>& ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    int n = ceil(Kw / Sw); // n is the number of ptr (NPO, PO2, PO3)
    int number = floor((Iw - Kw) / Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;

    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int* x_ptr = &ptr[0];
    int  x = *x_ptr;
    int* Aindex_help = &IN[x];
    int* Adata_help = &DA[x];

    // For each ptr type
    // for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    int type_ptr = 0;
    {


        // For each submat
        for (int submat = 0; submat < number; ++submat)
        {

            x = *x_ptr;
            int end_x_loop = *(x_ptr + 1);
            ++x_ptr;
            // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;


            for (; x < end_x_loop; ++x)
            {


                // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
                 // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

                 // How many time to iterate?
                int used_index = *Aindex_help; Aindex_help++;
                int used_data = *Adata_help;  Adata_help++;

                // int shereet2 = min(submat, type_ptr); 
                for (int l = 0; l < Kh; ++l)
                {
                    // I = 0:
                   // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                    int input_index = used_index;
                    int y_out = (input_index / Kw) - l;
                    int x_out = submat;

                    if (y_out >= 0 && y_out < Oh) {
                        O[y_out][x_out] += used_data * K[input_index % Kw + l * Kw];
                    } // end if(y_out >= 0 && y_out < Oh) 

                } // end l loop

            } // end x
        } // end sumbat

        ++x_ptr;
    } // end type ptr


      // For each ptr type
      // for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    type_ptr = 1;
    {


        // For each submat
        for (int submat = 0; submat < number; ++submat)
        {

            x = *x_ptr;
            int end_x_loop = *(x_ptr + 1);
            ++x_ptr;

            // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;


            for (; x < end_x_loop; ++x)
            {


                // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
                 // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

                 // How many time to iterate?
                int used_index = *Aindex_help; Aindex_help++;
                int used_data = *Adata_help;  Adata_help++;

                // int shereet2 = min(submat, type_ptr); 
                for (int l = 0; l < Kh; ++l)
                {
                    // I = 0:
                    int i = 0;
                    // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                    int input_index = used_index - i;
                    int y_out = (input_index / Kw) - l;
                    int x_out = i + submat;

                    if (y_out >= 0 && y_out < Oh) {
                        O[y_out][x_out] += used_data * K[input_index % Kw + l * Kw];

                        int input_index = used_index - 1;
                        O[y_out][x_out + 1] += used_data * K[input_index % Kw + l * Kw];

                    } // end if(y_out >= 0 && y_out < Oh) 

                } // end l loop

            } // end x
        } // end sumbat

        ++x_ptr;
    } // end type ptr

      // For each ptr type
      // for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    type_ptr = 2;
    {

        // For each submat
        for (int submat = 0; submat < number; ++submat)
        {

            x = *x_ptr;
            int end_x_loop = *(x_ptr + 1);
            ++x_ptr;

            // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;

            for (; x < end_x_loop; ++x)
            {


                // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
                 // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

                 // How many time to iterate?
                int used_index = *Aindex_help; Aindex_help++;
                int used_data = *Adata_help;  Adata_help++;

                // int shereet2 = min(submat, type_ptr); 
                for (int l = 0; l < Kh; ++l)
                {
                    // I = 0:
                    int i = 0;
                    // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                    int input_index = used_index - i;
                    int y_out = (input_index / Kw) - l;
                    int x_out = i + submat;

                    if (y_out >= 0 && y_out < Oh) {
                        O[y_out][x_out] += used_data * K[input_index % Kw + l * Kw];

                        int input_index = used_index - 1;
                        O[y_out][x_out + 1] += used_data * K[input_index % Kw + l * Kw];

                        i = 2;
                        input_index = used_index - 2;
                        O[y_out][x_out + 2] += used_data * K[input_index % Kw + l * Kw];

                    } // end if(y_out >= 0 && y_out < Oh) 

                } // end l loop

            } // end x
        } // end sumbat

        ++x_ptr;
    } // end type ptr

      // cout << "Output: " << endl;
      // print2DVectorF(O);
      // cout << "-----\n" << endl;
}

void conv_CPO_v5(vector<vector<float> >& O, vector<int> const& K, vector<int>& IN, vector<int>& DA, vector<int>& ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    int n = ceil(Kw / Sw); // n is the number of ptr (NPO, PO2, PO3)
    int number = floor((Iw - Kw) / Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;

    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int* x_ptr = &ptr[0];
    int  x = *x_ptr;
    int* Aindex_help = &IN[x];
    int* Adata_help = &DA[x];

    // For each ptr type
    for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    {


        // For each submat
        for (int submat = 0; submat < number; ++submat)
        {

            x = *x_ptr;
            int end_x_loop = *(x_ptr + 1);
            ++x_ptr;

            // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;


            for (; x < end_x_loop; ++x)
            {


                // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
                 // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

                 // How many time to iterate?
                int used_index = *Aindex_help; Aindex_help++;
                int used_data = *Adata_help;  Adata_help++;

                // int shereet2 = min(submat, type_ptr); 
                for (int l = 0; l < Kh; ++l)
                {
                    // I = 0:
                    int i = 0;
                    // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                    int input_index = used_index - i;
                    int y_out = (input_index / Kw) - l;
                    int x_out = i + submat;

                    bool flag = true;

                    if (y_out >= 0 && y_out < Oh) {
                        O[y_out][x_out] += used_data * K[input_index % Kw + l * Kw];

                        for (int i = 1; i <= type_ptr; ++i)
                        {
                            int input_index = used_index - i;
                            O[y_out][x_out + i] += used_data * K[input_index % Kw + l * Kw];

                        } // end i for loop
                    } // end if(y_out >= 0 && y_out < Oh) 

                } // end l loop

            } // end x
        } // end sumbat

        ++x_ptr;
    } // end type ptr

      // cout << "Output: " << endl;
      // print2DVectorF(O);
      // cout << "-----\n" << endl;
}


void conv_CPO_v3(vector<vector<float> >& O, vector<int> const& K, vector<int>& IN, vector<int>& DA, vector<int>& ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    int n = ceil(Kw / Sw); // n is the number of ptr (NPO, PO2, PO3)
    int number = floor((Iw - Kw) / Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;

    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int* x_ptr = &ptr[0];
    int  x = *x_ptr;
    int* Aindex_help = &IN[x];
    int* Adata_help = &DA[x];

    // For each ptr type
    for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    {


        // For each submat
        for (int submat = 0; submat < number; ++submat)
        {

            x = *x_ptr;
            int end_x_loop = *(x_ptr + 1);
            ++x_ptr;

            // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;


            for (; x < end_x_loop; ++x)
            {


                // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
                 // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

                 // How many time to iterate?
                int used_index = *Aindex_help; Aindex_help++;
                int used_data = *Adata_help;  Adata_help++;

                // int shereet2 = min(submat, type_ptr); 
                for (int i = 0; i <= type_ptr; ++i)
                {
                    // Loop on Kh for the output
                    for (int l = 0; l < Kh; ++l)
                    {


                        // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
                        int input_index = used_index - i;
                        int y_out = (input_index / Kw) - l;
                        int x_out = i + submat;

                        if (y_out < 0 || y_out >= Oh) {
                            // cout << "continue YYY============\n" << endl;
                            continue;
                        }

                        // cout << "R) " << y_out << ", C) " << x_out << ", Data: " << DA[type_ptr][x] << ", Index: " << input_index  << ", ac_Index: " << IN[type_ptr][x] << endl;
                       //    // O(y_out, x_out) += DA[type_ptr][x] * 1.0;
                        O[y_out][x_out] += used_data * K[input_index % Kw + l * Kw];

                    } // for each l in Kh
                } // end i
            } // end x
        } // end sumbat

        ++x_ptr;
    } // end type ptr

      // cout << "Output: " << endl;
      // print2DVectorF(O);
      // cout << "-----\n" << endl;
}


void conv_CPO_v2(vector<vector<float> >& O, vector<int> const& K, vector<vector<int> >& IN, vector<vector<int> >& DA, vector<vector<int> >& ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;




    int n = ceil(Kw / Sw); // n is the number of ptr (NPO, PO2, PO3)
    int number = floor((Iw - Kw) / Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;

    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";


    // For each ptr type
    // int type_ptr = 0;
    for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    {

        int x = ptr[type_ptr][0];
        int* Aindex_help = &IN[type_ptr][x];
        int* Adata_help = &DA[type_ptr][x];

        // For each submat
        for (int submat = 0; submat < number; ++submat)
        {

            // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << endl;
            for (; x < ptr[type_ptr][submat + 1]; ++x)
            {
                // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  

                 // How many time to iterate?
                int used_index = *Aindex_help; Aindex_help++;
                int used_data = *Adata_help; Adata_help++;

                // int shereet2 = min(submat, type_ptr); 
                for (int i = 0; i <= type_ptr; ++i)
                {

                    // cout << "Use x:  " << i << ", with ind: " << used_index << endl;

                    // Loop on Kh for the output
                    for (int l = 0; l < Kh; ++l)
                    {
                        int input_index = used_index - i;
                        int y_out = (input_index) / Kw - l;
                        int x_out = i + submat;

                        if (y_out < 0 || y_out >= Oh) {
                            // cout << "continue YYY============\n" << endl;
                            continue;
                        }


                        // cout << "R) " << y_out << ", C) " << x_out << ", Data: " << DA[type_ptr][x] << ", Index: " << input_index  << ", ac_Index: " << IN[type_ptr][x] << endl;
                       //    // O(y_out, x_out) += DA[type_ptr][x] * 1.0;
                        O[y_out][x_out] += used_data * K[input_index % Kw + l * Kw];

                    } // for each l in Kh
                } // end i
            } // end x
        } // end sumbat
    } // end type ptr

      // cout << "Output: " << endl;
      // print2DVectorF(O);
      // cout << "-----\n" << endl;
}

void conv_CPO_v1(vector<vector<float> >& O, vector<int> const& K, vector<vector<int> > const& IN, vector<vector<int> > const& DA, vector<vector<int> > const& ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    int n = ceil(Kw / Sw); // n is the number of ptr (NPO, PO2, PO3)
    int number = floor((Iw - Kw) / Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;

    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";


    // For each ptr type
    // int type_ptr = 0;
    for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    {

        // For each submat
        for (int submat = 0; submat < number; ++submat)
        {

            // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << endl;
            for (int x = ptr[type_ptr][submat]; x < ptr[type_ptr][submat + 1]; ++x)
            {
                // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  

                 // How many time to iterate?
                int used_index = IN[type_ptr][x];
                int used_data = DA[type_ptr][x];


                for (int i = 0; i <= type_ptr; ++i)
                {

                    // cout << "Use x:  " << i << ", with ind: " << used_index << endl;

                    // Loop on Kh for the output
                    for (int l = 0; l < Kh; ++l)
                    {
                        int input_index = used_index - i;
                        int y_out = (input_index) / Kw - l;
                        int x_out = submat + i;

                        if (y_out < 0 || y_out >= Oh) {
                            // cout << "continue YYY============\n" << endl;
                            continue;
                        }


                        // cout << "R) " << y_out << ", C) " << x_out << ", Data: " << DA[type_ptr][x] << ", Index: " << input_index  << ", ac_Index: " << IN[type_ptr][x] << endl;
                       //    // O(y_out, x_out) += DA[type_ptr][x] * 1.0;
                        O[y_out][x_out] += used_data * K[input_index % Kw + l * Kw];

                    } // for each l in Kh
                } // end i
            } // end x
        } // end sumbat
    } // end type ptr

      // cout << "Output: " << endl;
      // print2DVectorF(O);
      // cout << "-----\n" << endl;
}





void conv_CPO(vector<vector<float> >& O, vector<int> const& K, vector<vector<int> > const& IN, vector<vector<int> > const& DA, vector<vector<int> > const& ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    int n = ceil(Kw / Sw);
    int number = floor((Iw - Kw) / Sw) + 1;
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
            for (int i = 0; i <= shereet2; ++i)
            {

                // From ptr r t r+1
                int shereet = (type_ptr > 0) ? 1 : 0;

                for (int x = ptr[type_ptr][submat - i * shereet]; x < ptr[type_ptr][submat - i * shereet + 1]; ++x)
                {

                    // Loop on Kh for the output
                    for (int l = 0; l < Kh; ++l)
                    {
                        int input_index = IN[type_ptr][x] - i;
                        int y_out = (input_index) / Kw - l;
                        int x_out = submat;

                        if (y_out < 0 || y_out >= Oh) {
                            // cout << "continue YYY============\n" << endl;
                            continue;
                        }


                        // cout << "R) " << y_out << ", C) " << x_out << ", Data: " << DA[type_ptr][x] << ", Index: " << input_index  << ", ac_Index: " << IN[type_ptr][x] << endl;

                       // O(y_out, x_out) += DA[type_ptr][x] * 1.0;
                        O[y_out][x_out] += DA[type_ptr][x] * K[input_index % Kw + l * Kw];

                    } // for each l in Kh

                } // for each raga3 el shereet
            } // for each x from range ptr and ptr + 1
        } // for each submat

    }

    // cout << "Output: " << endl;
    // print2DVectorF(O);
    // cout << "-----\n" << endl;
}



void CPO(MatrixXf& lowered_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> >& IN,
    vector<vector<int> >& DA, vector<vector<int> >& ptr)
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
            int NZE_data = Adata[x];
            for (int l = 0; l < Kh; ++l)
            {
                int m = NZE_index / Kw - l;
                int Kindex = NZE_index % Kw + l * Kw;
                if (m < 0 || m >= Oh) continue;

                // cout << "R) " << m << ", C) " << n << ", " << Kindex << endl;
                O(m, n) += NZE_data * K[Kindex];
            }
        }

    }
} // end mult

void csrMult_v2(MatrixXf& O, VectorXf& K, vector<double>& Adata, vector<int>& Aindices, vector<int>& Aindptr, int Kh, int Kw, int Oh, int Ow)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    int x = Aindptr[0];
    int* Aindex_help = &Aindices[x];
    double* Adata_help = &Adata[x];
    for (int n = 0; n < Ow; ++n)
    {
        for (; x < Aindptr[n + 1]; ++x)
        {
            double result = 0.0;
            int NZE_index = *Aindex_help; Aindex_help++;
            int NZE_data = *Adata_help; Adata_help++;
            for (int l = 0; l < Kh; ++l)
            {
                int m = NZE_index / Kw - l;
                int Kindex = NZE_index % Kw + l * Kw;
                if (m < 0 || m >= Oh) continue;

                // cout << "R) " << m << ", C) " << n << ", " << Kindex << endl;
                O(m, n) += NZE_data * K[Kindex];
            }
        }

    }
} // end mult

void csrMult_v3(VectorXd& Ax, VectorXd& x, vector<double>& Adata, vector<int>& Aindices, vector<int>& Aindptr)
{
    // This code assumes that the size of Ax is numRowsA.
    int dataIdx = Aindptr[0];
    int* Aindex = &Aindices[dataIdx];

    for (int j = 0; j < Aindptr.size(); ++j)
    {
        cout << j << ") " << Aindptr[j] << endl;
    }

    cout << "\nCaught the exception 22222 " << endl;
    cout << "value of i: " << 0 << ", Ax Size: " << Ax.size() << endl;
    cout << "Loop AindPtr " << Aindptr[1] << ", AindPtr.size: " << Aindptr.size() << endl;
    cout << "dataIdx: " << dataIdx << endl;
    cout << "Adatat size " << Adata.size() << endl;
    for (int i = 0; i < Ax.size(); i++)
    {
        double Ax_i = 0.0;
        for (; dataIdx < Aindptr[i + 1]; dataIdx++)
        {
            if (i > Aindptr.size())
            {
                cout << "Go beyond Aindptr.size " << Aindptr.size() << ", " << Ax.size() << endl;
                exit(0);
            }
            if (dataIdx > Adata.size())
            {
                cout << "\nCaught the exception " << endl;
                cout << "value of i: " << i << ", Ax Size: " << Ax.size() << endl;
                cout << "Loop AindPtr " << Aindptr[i] << ", AindPtr.size: " << Aindptr.size() << endl;
                cout << "dataIdx: " << dataIdx << endl;
                cout << "Adatat size " << Adata.size() << endl;
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
            for (int l = 0; l < Kh; ++l)
            {
                int m = Aindices[x] / Kw - l;
                int Kindex = Aindices[x] % Kw + l * Kw;
                if (m < 0 || m >= Oh) continue;

                // cout << "R) " << m << ", C) " << n << ", " << Kindex << endl;
                O(m, n) += Adata[x] * K[Kindex];
            }
        }

    }
} // end mult


int main()
{

    // density:
    // float density = 1;
    // float density = 1; 
    float density = 0.05;


    // std::vector<int> I_list = {8,17,50};
    // std::vector<int> Kh_list = {3, 7, 1};
    // std::vector<int> Kw_list = {3, 1, 7};

    std::vector<int> I_list = { 8,17,50 };
    std::vector<int> Kh_list = { 3, 1, 3, 7, 1 };
    std::vector<int> Kw_list = { 3, 3, 1, 1, 7 };


    // std::vector<int> I_list = {17};
    // std::vector<int> Kh_list = {1};
    // std::vector<int> Kw_list = {7};


    for (; density < 1.05; density += 0.05)
    {


        // timer for im2col, csr
        float t_im2col = 0;
        float t_csr = 0;
        float t_cpo = 0;

        // bench iterations
       // int bench_iterations = 100000;

        // int bench_iterations = 10;
        // int bench_iterations = 2;
        int bench_iterations = 1;

        // Conv parameters:
        int padding = 0;
        int stride = 1;
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

        for (int I : I_list)
        {
            int Ih = I;
            int Iw = I;

            for (int KK = 0; KK < Kh_list.size(); ++KK)
            {


                int Kh = Kh_list[KK];
                int Kw = Kw_list[KK];

                // usleess case skip it
                if (Ih == 8)
                    if (Kh == 1 || Kh == 7)
                        continue;

                // int Ih = 35;
                // int Iw = 35;

                // int Ih = 8;
                // int Iw = 8;


                // Kernel dimensions
                // int Kh = 3;
                // int Kw = 3;

                // int Kh = 7;
                // int Kw = 1;

                // int Kh = 1;
                // int Kw = 7;

                // int Kh = 5;
                // int Kw = 5;


                // adjust the iterations based on Ih  
                if (Ih > 100)
                {
                    bench_iterations = 1000;
                }

                // int Ic = 32; // this is for the node
                int Ic = 1; // put it as articial for now
                int In = 1;

                int K = 1; // number of filters

                int Oh = (1 + Ih - Kh + 2 * padding) / stride; // removed + 1
                int Ow = (1 + Iw - Kw + 2 * padding) / stride;

                int iter = 1;  // total number of times to perform the test for each of dense, sparse multiplication

                // Create your original input feature map:
                MatrixXf org_fm = MatrixXf::Zero(Ih, Iw);

                // for(int i = 0; i < density*Ih*Iw; ++i)
                // {
                //   int r        = rand()%Ih;
                //   int c        = rand()%Iw;
                //   org_fm(r, c) = 1;
                // }

                // double sparsity_val{ 1.0 - density };
                // boost::random::uniform_01<> dist;
                // boost::random::mt19937 gen;
                // double density_cal = 0;

                // for (unsigned i = 0; i < Ih; ++i)
                // {
                //     for (unsigned j = 0; j < Iw; ++j)
                //     {   
                //         if (dist(gen)>sparsity_val){
                //           // org_fm(i, j) = dist(gen);
                //           org_fm(i, j) = 1;
                //           density_cal += 1;
                //         } 
                //     }
                // }
                // density_cal = density_cal/(Ih*Iw);
                // cout << "Calculated density: " << density_cal << endl;
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

                // std::cout << "\n===Original Feature Map (" << Ih << "x" << Iw <<  "):  \n" << org_fm << std::endl;
                // cout << "-----\n" << endl;

                // Create the filter K and its vectorized version:
                MatrixXf filter = MatrixXf::Ones(Kh, Kw);
                VectorXf filter_vectorized = VectorXf::Ones(Kh * Kw);

                // std::cout << "\n===Filter: " <<  " \n" << filter_vectorized  << std::endl;
                // cout << "-----\n" << endl;

                // Prepare the output for im2col, sparseMat
                MatrixXf d_o1 = MatrixXf::Zero(Oh, Ow);
                MatrixXf d_o2 = MatrixXf::Zero(Oh, Ow);

                // Create the Kernel
                vector<int> Kernel(Kh * Kw, 1);
                Kernel[2] = 2;



                // Prepare the Adata, Aindices, AindPtr for CPO multiplication
                int n = ceil(Kw / Sw);
                if (Kw % Sw == 0)
                {
                    n = Kw / Sw;
                }

                vector<vector<int> > IN(n); // n is the rows
                vector<vector<int> > DA(n); // n is the rows
                vector<vector<int> > ptr(n, vector<int>(Ow + 1, 0));

                CPO(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr);

                // CPO with repeats until v5
                // transform 2d to 1d:
                int count_ptr = 0;
                for (int i = 0; i < ptr.size(); ++i)
                {

                    count_ptr += ptr[i].size();

                }

                int count_d = 0;
                for (int i = 0; i < DA.size(); ++i)
                {

                    count_d += DA[i].size();
                }

                std::vector<int> IN_1d(count_d, 0);
                std::vector<int> DA_1d(count_d, 0);
                std::vector<int> ptr_1d(count_ptr, 0);

                transform2dTo1d(IN, DA, ptr, IN_1d, DA_1d, ptr_1d);
                ///////////
                // CPO without repeats v7 onwards
                int count_ptr_v7 = 0;
                for (int i = 0; i < ptr.size(); ++i)
                {

                    // if it one ptr, don't do anything
                    if (i == 0 && n != 1)
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
                for (int i = 0; i < DA.size(); ++i)
                {

                    count_d_v7 += DA[i].size();
                }

                std::vector<int> IN_1d_v7(count_d_v7, 0);
                std::vector<int> DA_1d_v7(count_d_v7, 0);
                std::vector<int> ptr_1d_v7(count_ptr_v7, 0);

                transform2dTo1dv1(IN, DA, ptr, IN_1d_v7, DA_1d_v7, ptr_1d_v7);

                ///////////


                vector<vector<float> > OV7(Oh, vector<float>(Ow, 0));
                {

                    if (n != 1)
                    {
                        clock_t t;


                        // cout << ptr.size() << "\t" << ptr[0].size() << endl;
                        for (int k = 0; k < bench_iterations; k++) {

                            // Prepare the output for CPO           
                            t = clock();
                            // conv_CPO(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v1(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v2(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v3(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                            // conv_CPO_v7_trim(OV7, Kernel, IN_1d_v7,  DA_1d_v7, ptr_1d_v7, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                            conv_CPO_v8_trim(OV7, Kernel, IN_1d_v7, DA_1d_v7, ptr_1d_v7, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            double elapsed = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds
                            // t_cpo1+=elapsed/(Ih*Iw*1.0); // normalized timing
#if case_1
                            if (k == bench_iterations - 1)
                            {
                                 // cout << "\t\t\tCPO_org" << endl;
                                 // print2DVectorF(OV7);
                                 // cout << "-----\n" << endl;
                                 // cout << "ptr new \n";
                                 // printVector(ptr_1d_v7);
                                 // cout << "IN new \n";
                                 // printVector(IN_1d_v7);
                            }
#endif
                            // Reset in all except last
                            if (k != bench_iterations - 1 && bench_iterations != 1)
                                reset2DVectorF(OV7);

                        }

                    }// end if
                    else
                    {
                        // cout << "CPO: ptr v7 \n";
                        // printVector(ptr_1d);
                        // cout << "CPO: IN v7 \n";
                        // printVector(IN_1d);

                        clock_t t;


                        // cout << ptr.size() << "\t" << ptr[0].size() << endl;
                        for (int k = 0; k < bench_iterations; k++) {

                            // Prepare the output for CPO           
                            t = clock();
                            // conv_CPO(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v1(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v2(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v3(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                            conv_CPO_v7(OV7, Kernel, IN_1d, DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            double elapsed = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds
                            // t_cpo1+=elapsed/(Ih*Iw*1.0); // normalized timing
#if case_1
                            if (k == bench_iterations - 1)
                            {
                                 // cout << "\t\t\tCPO_org" << endl;
                                 // print2DVectorF(OV7);
                                 // cout << "-----\n" << endl;
                                 // cout << "ptr new \n";
                                 // printVector(ptr_1d_v7);
                                 // cout << "IN new \n";
                                 // printVector(IN_1d_v7);
                            }
#endif
                            // Reset in all except last
                            if (k != bench_iterations - 1 && bench_iterations != 1)
                                reset2DVectorF(OV7);

                            // if(k == bench_iterations-1)
                            // {
                            //  print2DVectorF(O);
                            //  cout << "-----\n" << endl;
                            // }
                        }
                    }
                }




#if 0
                vector<vector<float> > O_cpo(Oh, vector<float>(Ow, 0));
                {

                    if (n != 1)
                    {

                        count_ptr = n * (1 + Ow);
                        if (n != 1)
                        {
                            count_ptr = (n - 1) * (1 + Ow) + min(1 + Ow, 3);

                        }



                        std::vector<int> ptr_1d_enc(count_ptr, -1);
                        // std::vector<int> ptr_1d_enc(count_ptr, 0);

                        int m3_pos = Iw - 2 * Kw + 2;
                        std::vector<int> IN_1d_enc;
                        std::vector<int> DA_1d_enc;
                        vector<vector<int> > IN_enc(n); // n is the rows
                        vector<vector<int> > DA_enc(n); // n is the rows
                        // vector<vector<int> > ptr_enc(n, vector<int>(Ow + 1, 0));
                        std::vector<int> x(n, 0);
                        std::vector<int> m(n, m3_pos);


                        int count_d_enc = CPO4(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN_enc, DA_enc, ptr_1d_enc, x, m);

                        IN_1d.reserve(count_d_enc);
                        DA_1d.reserve(count_d_enc);

                        transform2dTo1d_encoding(IN_enc, DA_enc, IN_1d_enc, DA_1d_enc);

                        // cout << Iw << "\t" << Kw << "\t" << Kh << endl;

                        clock_t t;


                        // cout << ptr.size() << "\t" << ptr[0].size() << endl;
                        for (int k = 0; k < bench_iterations; k++) {

                            // Prepare the output for CPO           
                            t = clock();
                            // conv_CPO(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v1(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v2(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v3(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                            // conv_CPO_v7_trim(OV7, Kernel, IN_1d_v7,  DA_1d_v7, ptr_1d_v7, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                            conv_CPO_v8_trim(O_cpo, Kernel, IN_1d_enc, DA_1d_enc, ptr_1d_enc, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            double elapsed = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds
                            // t_cpo1+=elapsed/(Ih*Iw*1.0); // normalized timing

#if case_1
                            if (k == bench_iterations - 1)
                            {
                                 cout << "\t\t\tCPO_4: " << endl;
                                 print2DVectorF(O_cpo);
                                 cout << "-----\n" << endl;
                                 cout << "ptr new \n";
                                 printVector(ptr_1d_enc);
                                 cout << "IN new \n";
                                 printVector(IN_1d_enc);
                            }
#endif
                            // Reset in all except last
                            if (k != bench_iterations - 1 && bench_iterations != 1)
                                reset2DVectorF(OV7);

                            // if(k == bench_iterations-1)
                            // {
                            //  print2DVectorF(O);
                            //  cout << "-----\n" << endl;
                            // }
                        }

                    }// end if
                    else
                    {
                        clock_t t;


                        // cout << ptr.size() << "\t" << ptr[0].size() << endl;
                        std::vector<int> DA_1d_csr_d;
                        std::vector<int> IN_1d_csr;
                        std::vector<int> ptr_1d_csr;
                        CSR(DA_1d_csr_d, IN_1d_csr, ptr_1d_csr, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                        // cout << "CSR @Ahmad" << endl;
                        // printVector(ptr_1d_csr);
                        // printVector(IN_1d_csr);

                        for (int k = 0; k < bench_iterations; k++) {

                            // Prepare the output for CPO           
                            t = clock();
                            // conv_CPO(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v1(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v2(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v3(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                            // conv_CPO_v7(O_cpo, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);


                            conv_CPO_v7(O_cpo, Kernel, IN_1d_csr, DA_1d_csr_d, ptr_1d_csr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            double elapsed = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds
                            // t_cpo1+=elapsed/(Ih*Iw*1.0); // normalized timing
#if case_1
                            if (k == bench_iterations - 1)
                            {
                                 cout << "\t\t\tCPO_4: " << endl;
                                 print2DVectorF(O_cpo);
                                 cout << "-----\n" << endl;
                                 cout << "ptr new \n";
                                 printVector(ptr_1d_csr);
                                 cout << "IN new \n";
                                 printVector(IN_1d_csr);
                            }
#endif
                            // Reset in all except last
                            if (k != bench_iterations - 1 && bench_iterations != 1)
                                reset2DVectorF(OV7);

                            // if(k == bench_iterations-1)
                            // {
                            //  print2DVectorF(O);
                            //  cout << "-----\n" << endl;
                            // }
                        }
                    }
                }

#endif

                vector<vector<float> > O_cpo_v8(Oh, vector<float>(Ow, 0));
                {

                    if (n != 1)
                    {

                        count_ptr = n * (1 + Ow);
                        if (n != 1)
                        {
                            count_ptr = (n - 1) * (1 + Ow) + min(1 + Ow, 3);

                        }


                        std::vector<int> ptr_1d_enc_v8(count_ptr, -1);
                        // std::vector<int> ptr_1d_enc(count_ptr, 0);

                        int m3_pos = Iw - 2 * Kw + 2;
                        std::vector<int> x(n, 0);
                        std::vector<int> m(n, m3_pos);
                        for (int i = 1 ; i < m.size()-1 ; i++)
                        {
                            m[i] = Ow - i;
                        }
                        std::vector<int> IN_1d_enc_v8;
                        std::vector<int> DA_1d_enc_v8;
                        // vector<vector<int> > ptr_enc(n, vector<int>(Ow + 1, 0));

                        // int count_d_enc = CPO4(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN_enc, DA_enc, ptr_1d_enc, x, m);
                        CPO5_OP(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, ptr_1d_enc_v8, IN_1d_enc_v8, DA_1d_enc_v8, x, m);

                        clock_t t;


                        // cout << ptr.size() << "\t" << ptr[0].size() << endl;
                        for (int k = 0; k < bench_iterations; k++) {

                            // Prepare the output for CPO           
                            t = clock();
                            // conv_CPO(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v1(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v2(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v3(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                            // conv_CPO_v7_trim(OV7, Kernel, IN_1d_v7,  DA_1d_v7, ptr_1d_v7, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                            conv_CPO_v8_trim(O_cpo_v8, Kernel, IN_1d_enc_v8, DA_1d_enc_v8, ptr_1d_enc_v8, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            double elapsed = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds
                            // t_cpo1+=elapsed/(Ih*Iw*1.0); // normalized timing

#if case_1
                            if (k == bench_iterations - 1)
                            {
                                 // cout << "\t\t\tCPO_5: " << endl;
                                 // print2DVectorF(O_cpo_v8);
                                 // cout << "-----\n" << endl;
                                 // cout << "ptr new \n";
                                 // printVector(ptr_1d_enc_v8);
                                 // cout << "IN new \n";
                                 // printVector(IN_1d_enc_v8);
                            }
#endif
                            // Reset in all except last
                            if (k != bench_iterations - 1 && bench_iterations != 1)
                                reset2DVectorF(O_cpo_v8);

                            // if(k == bench_iterations-1)
                            // {
                            //  print2DVectorF(O);
                            //  cout << "-----\n" << endl;
                            // }
                        }

                    }// end if
                    else
                    {
                        clock_t t;


                        // cout << ptr.size() << "\t" << ptr[0].size() << endl;
                        std::vector<int> DA_1d_csr_d;
                        std::vector<int> IN_1d_csr;
                        std::vector<int> ptr_1d_csr;
                        CSR(DA_1d_csr_d, IN_1d_csr, ptr_1d_csr, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                        // cout << "CSR @Ahmad" << endl;
                        // printVector(ptr_1d_csr);
                        // printVector(IN_1d_csr);

                        for (int k = 0; k < bench_iterations; k++) {

                            // Prepare the output for CPO           
                            t = clock();
                            // conv_CPO(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v1(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v2(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            // conv_CPO_v3(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                            // conv_CPO_v7(O_cpo, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);


                            conv_CPO_v7(O_cpo_v8, Kernel, IN_1d_csr, DA_1d_csr_d, ptr_1d_csr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                            double elapsed = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds
                            // t_cpo1+=elapsed/(Ih*Iw*1.0); // normalized timing
#if case_1
                            if (k == bench_iterations - 1)
                            {
                                 // cout << "\t\t\tCPO_5: " << endl;
                                 // print2DVectorF(O_cpo_v8);
                                 // cout << "-----\n" << endl;
                                 // cout << "ptr new \n";
                                 // printVector(ptr_1d_csr);
                                 // cout << "IN new \n";
                                 // printVector(IN_1d_csr);
                            }
#endif
                            // Reset in all except last
                            if (k != bench_iterations - 1 && bench_iterations != 1)
                                reset2DVectorF(O_cpo_v8);

                            // if(k == bench_iterations-1)
                            // {
                            //  print2DVectorF(O);
                            //  cout << "-----\n" << endl;
                            // }
                        }
                    }
                }


#if 0
               vector<vector<float> > O_CPOV8( Oh , vector<float> (Ow, 0));
               {  

                    if(n != 1)
                    {

                        count_ptr = n * (1 + Ow);
                        if (n != 1)
                        {
                        count_ptr = (n - 1) * (1 + Ow) + min(1 + Ow, 3);

                        }



                        std::vector<int> ptr_1d_enc(count_ptr, -1);
                        // std::vector<int> ptr_1d_enc(count_ptr, 0);

                        int m3_pos = Iw - 2 * Kw + 2;
                        std::vector<int> x(n, 0);
                        std::vector<int> m(n, m3_pos);
                        for (int i = 1 ; i < m.size()-1 ; i++)
                        {
                            m[i] = Ow - 1;
                        }
                        std::vector<int> IN_1d_enc;
                        std::vector<int> DA_1d_enc;
                        // vector<vector<int> > ptr_enc(n, vector<int>(Ow + 1, 0));

                        // int count_d_enc = CPO4(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN_enc, DA_enc, ptr_1d_enc, x, m);
                        CPO5_OP(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, ptr_1d_enc, IN_1d_enc, DA_1d_enc, x, m);

                        // cout << Iw << "\t" << Kw << "\t" << Kh << endl;
                        std::vector<double> DA_1d_enc_d(DA_1d_enc.size(), 0);

                        for(int h = 0; h < DA_1d_enc.size(); ++h)
                        {
                          DA_1d_enc_d[h] = DA_1d_enc[h];
                        }
                        clock_t t;
                        
                    
                      // cout << ptr.size() << "\t" << ptr[0].size() << endl;
                        for(int k=0;k<bench_iterations;k++){

                           // Prepare the output for CPO           
                           t = clock();
                           // conv_CPO(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                           // conv_CPO_v1(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                           // conv_CPO_v2(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                           // conv_CPO_v3(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                           // conv_CPO_v7_trim(OV7, Kernel, IN_1d_v7,  DA_1d_v7, ptr_1d_v7, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                           // conv_CPO_v8_trim(O_CPOV8, Kernel_d, IN_1d_enc,  DA_1d_enc_d, ptr_1d_enc, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                           conv_CPO_v8_trim(O_CPOV8, Kernel, IN_1d_enc,  DA_1d_enc, ptr_1d_enc, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                           double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
                           // t_cpo1+=elapsed/(Ih*Iw*1.0); // normalized timing

#if case_1
                           if(k == bench_iterations-1)
                           {
                            cout << "\t\t\tCPO5" << endl;
                            cout << "ptr new \n";
                            printVector(ptr_1d_enc);
                            cout << "IN new \n";
                            printVector(IN_1d_enc);
                            cout << "\n\nCPO_v8_enc: " << endl;
                            print2DVectorF(O_CPOV8);
                            cout << "-----\n" << endl;
                            }
#endif
                           // Reset in all except last
                           if(k != bench_iterations - 1 && bench_iterations != 1)
                            {

                                reset2DVectorF(O_CPOV8);
                            }
                           // if(k == bench_iterations-1)
                           // {
                           //  print2DVectorF(O);
                           //  cout << "-----\n" << endl;
                           // }
                       }

                 }// end if
                 else
                 {
                    clock_t t;
                    
                
                  // cout << ptr.size() << "\t" << ptr[0].size() << endl;
                    std::vector<int> DA_1d_csr_d;
                    std::vector<int> IN_1d_csr;
                    std::vector<int> ptr_1d_csr;
                    CSR(DA_1d_csr_d, IN_1d_csr, ptr_1d_csr, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                    // cout << "CSR @Ahmad" << endl;
                    // printVector(ptr_1d_csr);
                    // printVector(IN_1d_csr);

                   for(int k=0;k<bench_iterations;k++){

                       // Prepare the output for CPO           
                       t = clock();
                       // conv_CPO(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                       // conv_CPO_v1(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                       // conv_CPO_v2(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                       // conv_CPO_v3(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                       // conv_CPO_v7(O_cpo, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                       
                       conv_CPO_v7(O_CPOV8, Kernel, IN_1d_csr,  DA_1d_csr_d, ptr_1d_csr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                       double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
                       // t_cpo1+=elapsed/(Ih*Iw*1.0); // normalized timing
#if case_1
                       if(k == bench_iterations-1)
                       {
                        cout << "CPO V8: " << endl;
                        print2DVectorF(O_CPOV8);
                        cout << "-----\n" << endl;
                       }
#endif
                       // Reset in all except last
                       if(k != bench_iterations - 1 && bench_iterations != 1)
                        { 

                            reset2DVectorF(O_CPOV8);
                        }
                       // if(k == bench_iterations-1)
                       // {
                       //  print2DVectorF(O);
                       //  cout << "-----\n" << endl;
                       // }
                   }
                 }
               } 

#endif
                // CSCC
#if 0 
                // Create the lowered matrix:
                MatrixXf lowered_mat = MatrixXf::Zero(Ow, Ih * Kw);
                int sub_matrix_index = 0;
                // For each submatrix:
                for (int sub_m_start_col = 0; sub_m_start_col < Ow; sub_m_start_col = sub_m_start_col + stride)
                {

                    if (sub_m_start_col + Kw > Iw)
                    {
                        break;
                    }

                    int lowered_mat_col_index = 0;
                    // Fetch this piece (all rows, all cols in this current submatrix)
                    for (int row_int = 0; row_int < Ih; ++row_int)
                    {
                        for (int col_int = sub_m_start_col; col_int < sub_m_start_col + Kw; ++col_int)
                        {
                            // cout << (org_fm(row_int, col_int)) << ", ";     

                            lowered_mat(sub_matrix_index, lowered_mat_col_index) = org_fm(row_int, col_int);
                            lowered_mat_col_index++;
                        } // end inner loop

                    } // end outer loop

                    sub_matrix_index++;

                } // end outer outer loop

                
                // Create the sparse representation of the lowered matrix:
                SparseMatrix<float, RowMajor> lowered_mat_sparse = lowered_mat.sparseView();
                lowered_mat_sparse.makeCompressed();

                // Prepare the Adata, Aindices, AindPtr for CSR multiplication
                int nz = lowered_mat_sparse.nonZeros();
                vector<int> Adata(lowered_mat_sparse.valuePtr(), lowered_mat_sparse.valuePtr() + nz);
                vector<int> Aindices(lowered_mat_sparse.innerIndexPtr(), lowered_mat_sparse.innerIndexPtr() + nz);
                vector<int> Aindptr(lowered_mat_sparse.outerIndexPtr(), lowered_mat_sparse.outerIndexPtr() + lowered_mat_sparse.outerSize()); // +1 for the last element
                // push back the last element the number of nnz in ptr:
                Aindptr.push_back(nz);
#else
                std::vector<int> Adata;
                std::vector<int> Aindices;
                std::vector<int> Aindptr;
                CSR(Adata, Aindices, Aindptr, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
#endif           
                // Perform 50 times raw sparse matrix dense vector multiplication: d_o2 = d_m * d_b [Without Eigen]
                vector<vector<float> > O_CSR(Oh, vector<float>(Ow, 0));
                {
                    clock_t t;
                    for (int k = 0; k < bench_iterations; k++) {
                        // Prepare the output for CSR
                        t = clock();
                        // csrMult_v2(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
                        
                        csrMult_v4(O_CSR, Kernel, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
                        double elapsed = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds 
                        t_csr += elapsed / (Ih * Iw * 1.0); // normalized timing



                        if (k == bench_iterations - 1)
                        {
                            // cout << " CSR Eigen \n";
                            // printVector(Aindptr);
                            // printVector(Aindices);
                              // cout << "CSR without Eigen Output: " << endl;
                              // print2DVectorF(O_CSR);
                              // cout << "-----\n" << endl;
                        }


                        if (k != bench_iterations - 1 && bench_iterations != 1)
                            reset2DVectorF(O_CSR);

                    }
                }

                // For V7 and V8:
                // bool error = isEqualVectors(O_CSR, OV7);
                // bool error = isEqualVectors(O_CSR, O_CPOV8);
                // bool error = isEqualVectors(O_CSR, O_cpo); 
                bool error = isEqualVectors(O_CSR, O_cpo_v8);

                // For V9:
                // bool error = isEqualVectors(O_CSR, OV9);

                // print2DVectorF(O);
                // print2DVectorF(O_CSR);
                if (!error)
                    std::cout << Kh << "x" << Kw << " | " << Ih << "x" << Iw << ") batch\t" << 1
                    << "\ttarget_density\t" << density << "\tdensity\t" << density_cal
                    << "\tverify:\t" << std::boolalpha << !error << "\n";
                else
                    std::cout << "XXXXXXXXXXX\t" << Kh << "x" << Kw << " | " << Ih << "x" << Iw << ") batch\t" << 1
                    << "\ttarget_density\t" << density << "\tdensity\t" << density_cal
                    << "\tverify:\t" << std::boolalpha << !error << "\n";



                // O_CPOV8.resize(0);
                O_CSR.resize(0);
                OV7.resize(0);
                // O_cpo.resize(0);


                // elapsed time per feature element in the entire bench iterations
                // std::cout<<"batch\t"<<In<<"\tdensity\t"<<density <<"\tdensity_cal\t"<<density_cal <<"\tim2col\t"<< t_im2col <<"\tcsr\t"<< t_csr <<"\tcpo\t"<< t_cpo <<std::endl;
                  // std::cout << "CPO:\t" << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batch\t"<<1
                  //     <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
                  //     <<"\tim2col\t"<<t_im2col<<"\tcsr\t" <<t_csr <<"\tcpo\t"<< t_cpo
                  //      <<"\tpercent1\t"<< 100.0*(t_im2col-t_csr)/t_im2col  <<"\tpercent2\t"<< 100.0*(t_im2col-t_cpo)/t_im2col << "\n";


            } // end K list loop
        } // end I list loop


    } // density loop

    cout << "Smile All Done!" << endl;

    return 0;
}