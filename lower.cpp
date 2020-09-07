#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <boost/timer/timer.hpp>
#include <time.h>
#include <math.h>


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
    o.noalias() = m*in;
    
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
    for(int i = 0; i < x.size(); ++i)
    {
        cout << x[i] << ", ";
    }
    cout << "\n";
}

void _CPO(MatrixXf& O, VectorXf& K, MatrixXf& lowered_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
    std::cout << "\n===Lowered Feature Map of Size: " << lowered_mat.rows() << ", " << lowered_mat.cols() <<  "\n" << lowered_mat << std::endl;
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
    
    int count = 0;
    int interm_count = 0;
    std::vector<int> seq_ptr;
    std::vector<int> ptrs;
    std::vector<int> Interm;
    
    for(int c = 0; c < floor(Kw/Sw); ++c)
    {
        //seq_ptr[count] = Sw;
        //ptr[count]   = count;
        //Interm[count] = interm_count;
        
        seq_ptr.push_back(Sw);
        ptrs.push_back(count);
        Interm.push_back(interm_count);
        count++;
    }
    if((Kw%Sw) != 0)
    {
        seq_ptr.push_back(Kw%Sw);
        ptrs.push_back(count);
        Interm.push_back(interm_count);
        count++;
    }
    
    
    interm_count++;
    for (int c = 0; c < floor((Iw-(2*Kw))/Sw); ++c)
    {
        seq_ptr.push_back(Sw-(Kw%Sw));
        seq_ptr.push_back(Kw%Sw);
        
        ptrs.push_back(floor(Kw/Sw)-1);
        ptrs.push_back(floor(Kw/Sw));
        
        Interm.push_back(interm_count);
        Interm.push_back(interm_count);
        count += 2;
        interm_count++;
    }
    
    
    if(floor((Iw-(2*Kw))/Sw) != 0)
    {
        seq_ptr.push_back(Sw-(Kw%Sw));
        ptrs.push_back(floor(Kw/Sw) - 1);
        Interm.push_back(interm_count);
        count++;
    }
    
    if((Kw%Sw) != 0)
    {
        seq_ptr.push_back(Kw%Sw);
        ptrs.push_back(floor(Kw/Sw));
        Interm.push_back(interm_count);
        count++;
    }
    
    interm_count++;
    
    cout <<  "Seq ptr: " << seq_ptr.size() << endl;
    
    for(int c = 0; c < int(floor(Kw/Sw)); ++c)
    {
        // cout << c << ", " << seq_ptr.size() << endl;
        //seq_ptr[count] = Sw;
        //ptr[count]   = count;
        //Interm[count] = interm_count;
        
        seq_ptr.push_back(Sw);
        ptrs.push_back(floor(Kw/Sw)-(c+1));
        Interm.push_back(interm_count);
        count++;
        interm_count++;
    }
    
    cout << "Seq ptr: " << seq_ptr.size() << endl;
    printVector(seq_ptr);
    exit(0);
    
    int j = 0; int num_nnz = 0;
    vector<vector<int> > Ptr(n); // n is the rows
    vector<vector<int> > DN(n); // n is the rows
    vector<vector<int> > IN(n); // n is the rows
    
    // Ptr declaration
    for(int p = 0; p < n; ++p)
    {
        Ptr[p] = vector<int>(Ow + 1, 0);
    }
    
    for(int i_  = 0; i_ < seq_ptr.size() - 1; ++i_)
    {
        cout << "\ni_: " << i_ << endl;
        for(int j_ = 0; j_ < seq_ptr[i]; ++j_)
        {
            cout << "j: " << j_ << endl;
            for(int i = 0; i < Ih; ++i)
            {
                cout << "i: " << i << endl;
                if(lowered_mat(i, j) != 0)
                {
                    IN[ptrs[i_]].push_back( (j + (i*Kw))-(Interm[i_]*Sw) );
                    DN[ptrs[i_]].push_back( lowered_mat(i, j) );
                    num_nnz++;
                }
            } // end loop 1
            j++;
        }// end loop 2
        
        for(int ptr_c = 0; ptr_c < n; ++ptr_c)
        {
            Ptr[ptr_c][Interm[i_]+1] = Ptr[ptr_c][Ptr.size()-1]; // --?
        }
        
        Ptr[ptrs[i_]][Interm[i_]+1]  = num_nnz + Ptr[ptrs[i_]][Interm[i_]+1];
        num_nnz = 0;
        
    }// end loop 3
    
    print2DVector(Ptr);
    
    exit(0);
}

void CPO(MatrixXf& O, VectorXf& K, MatrixXf& lowered_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
    std::cout << "\n===Lowered Feature Map of Size: " << lowered_mat.rows() << ", " << lowered_mat.cols() <<  "\n" << lowered_mat << std::endl;
    
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
    vector<vector<int> > IN(n); // n is the rows
    vector<vector<int> > DA(n); // n is the rows
    vector<vector<int> > ptr(n); // n is the rows
    
    // Ptr declaration
    for(int p = 0; p < n; ++p)
    {
        ptr[p] = vector<int>(Ow + 1);
    }
    
    // First piece
    for (int j = 0; j < Kw; ++j)
    {
        for(i = 0; i < Ih; ++i)
        {
            if (lowered_mat(i, j) != 0)
            {
                // cout << i << ", " << j << endl;
                IN[l].push_back(j + (i*Kw));
                DA[l].push_back(lowered_mat(i, j));
                
                if (flag == 0)
                {
                    ptr[l][m[l]] = x[l];
                    flag = 1;
                    m[l]++;
                } // end if (flag == 0)
                
                x[l]++;
            } // end if  if (lowered_mat(i, j) != 0)
            
        } // end for(i=0; i < Ih; ++i)
        
        if ( (j+1) % Sw == 0)
        {
            ptr[l][m[l]] = x[l];
            l++;
            flag = 0;
        } // end if ( (j+1) % Sw == 0)
        else if(j == Kw - 1)
        {
            ptr[l][m[l]] = x[l];
        } // end if(j == Kw - 1)
        
    } // end for (int j = 0; j < Kw; ++j)
    
    
    l--;
    
    for(int p = 0; p < m.size(); ++p)
    {
        m[p]++;
    }
    flag = 0;
    
    
    // Second piece
    for(int j = Kw; j < Iw - Kw; ++j)
    {
        for(i = 0; i < Ih; ++i)
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
        
        if(flag == 0)
        {
            if (Kw % Sw == 0)
            {
                if((j - Kw + 1) % Sw == 0)
                {
                    ptr[l][m[l]] = x[l];
                    m[l]++;
                    
                    if(n > 1)
                    {
                        for(int c = 0; c < n-1; ++c)
                        {
                            ptr[c][m[c]] = x[c];
                            m[c]++;
                        } // end for(int c = 0; c < n-1; ++c)
                    } // end if(n > 1)
                } // end if ((j - Kw + 1) % Sw == 0)
            } // end if (Kw % Sw == 0)
            else if ( (j - Kw + 1) % Sw == (Sw - (Kw % Sw)))
            {
                ptr[l][m[l]] = x[l];
                m[l]++;
                l++;
                flag = 1;
            } // end else if ( (j - Kw + 1) % Sw == (Sw - (Kw % Sw)))
        } // end if(flag == 0)
        
        else
        {
            if ( (j - Kw + 1) % Sw == 0)
            {
                ptr[l][m[l]] = x[l];
                m[l]++;
                l--;
                flag = 0;
                
            } // end if ( (j - Kw + 1) % Sw == 0)
            if(n > 2)
            {
                for(int c = 0; c < n-2; ++c)
                {
                    ptr[c][m[c]] = x[c];
                    m[c]++;
                } // end for(int c = 0; c < n-2; ++c)
            } // end if n > 2
        } // end if flag == 1
        
    } // end for(int j = Kw; j < Iw - Kw; ++j)
    
    
    
    // Third piece
    flag = 1;
    i = 0;
    for (int j = Iw < Kw; Iw; ++j)
    {
        for(int i = 0; i < Ih; i++)
        {
            cout << "2222  Row: " << i << ", Col: " << j << endl;
            
            if(lowered_mat(i, j) != 0)
            {
                IN[l].push_back(j + (i*Kw) - (Sw*(m[l] - 1)));
                DA[l].push_back(lowered_mat(i, j));
                x[l]++;
            }// end if(lowered_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)
        
        if((Iw - j - 1) % Sw == 0)
        {
            for(int c = 0; c < l+1; ++c)
            {
                ptr[c][m[c]] = x[c];
                m[c]++;
            } // end for(int c = 0; c < l+1; ++c)
            
            if(l > 1)
            {
                for(int c = 0; c < l-1; ++c)
                {
                    ptr[c][m[c]] = x[c];
                    m[c]++;
                } // end for(int c = 0; c < l-1; ++c)
            }// end if l > 1
            
            else if(l == 1)
            {
                ptr[0][m[0]] = x[0];
                m[0]++;
            } // end else if(l == 1)
            
            l--;
        } // end if ((Iw - j - 1) % Sw == 0)
        
        if( l < 0)
        {
            break;
        }
    } // end for (j = Iw - Kw; Iw; ++j)
    
    cout << "Done processing " << endl;
    print2DVector(ptr);
    
    exit(0);
}

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

void csrMult_v1(MatrixXf& O, VectorXf& K, vector<double>& Adata, vector<int>& Aindices, vector<int>& Aindptr, int Kh, int Kw, int Oh, int Ow)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;
    
    int x                = Aindptr[0];
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



// https://web.stanford.edu/class/cs315b/projects/conjugate_gradient/conjugate_gradient.cpp
// Parallel version
void csrMult_vp(MatrixXf& O, VectorXf& K, vector<double>& Adata, vector<int>& Aindices, vector<int>& Aindptr, int Kh, int Kw, int Oh, int Ow, int start_row,
                int end_row)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;
    
    int n, x, l, m, Kindex;
    
#pragma omp parallel num_threads(2)
    {
#pragma omp parallel for default(none) private(n, x, l, m, Kindex) shared(O, Adata, Aindices, Aindptr, K, Kh, Kw, Oh, Ow)
        for (n = 0; n < Ow/2; ++n)
        {
            for (x = Aindptr[n]; x < Aindptr[n + 1]; ++x)
            {
                for(l = 0; l < Kh; ++l)
                {
                    m      = Aindices[x]/Kw - l;
                    Kindex = Aindices[x]%Kw + l*Kw;
                    if(m < 0 || m >= Oh) continue;
                    
                    // cout << "R) " << m << ", C) " << n << ", " << Kindex << endl;
                    O(m, n) += Adata[x]*K[Kindex];
                }
            }
            
        }
    }
} // end mult

int main()
{
    
    // density:
    float density = 0.1;
    
    // timer for im2col, csr
    float t_im2col = 0;
    float t_csr    = 0;
    
    // bench iterations
    int bench_iterations = 100000;
    
    
    // Conv parameters:
    int padding = 0;
    int stride  = 1;
    int Sh, Sw;
    Sh = Sw = stride;
    int num_filters = 1; // 64
    
    // mixed 0: conv_node 3
    int Ih = 5;
    int Iw = 5;
    // int Ic = 32; // this is for the node
    int Ic = 1; // put it as articial for now
    int In = 1;
    
    int K = 1; // number of filters
    
    // Kernel dimensions
    int Kh = 3;
    int Kw = 3;
    
    int Oh = (1 + Ih - Kh + 2 * padding)/stride; // removed + 1
    int Ow = (1 + Iw - Kw + 2 * padding)/stride;
    
    int iter = 1;  // total number of times to perform the test for each of dense, sparse multiplication
    
    // Create your original input feature map:
    MatrixXf org_fm = MatrixXf::Zero(Ih, Iw);
    
    std::vector<int> cols = {0,1,4,0,4,0,4};
    std::vector<int> rows = {0,0,0,2,2,3,3};
    std::vector<double> values = {1,1,1,1,1,1,1};
    
    for(int i=0; i < cols.size(); i++)
    {
        org_fm(rows[i], cols[i])        = values[i];
    }
    
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
    
    // Create the filter K and its vectorized version:
    MatrixXf filter             = MatrixXf::Ones(Kh, Kw);
    VectorXf filter_vectorized  = VectorXf::Ones(Kh*Kw);
    
    // Print out the im2col interedmiate feature map:
    std::cout << "\n===Filter: " <<  " \n" << filter_vectorized  << std::endl;
    cout << "-----\n" << endl;
    
    // Prepare the output for im2col, sparseMat, CPO
    MatrixXf d_o1 = MatrixXf::Zero(Oh, Ow);
    MatrixXf d_o2 = MatrixXf::Zero(Oh, Ow);
    MatrixXf d_o3 = MatrixXf::Zero(Oh, Ow);
    
    // transpose the matrix for im2col:
    MatrixXf im2col_mat_tr = im2col_mat.transpose();
    
    // Perform 50 times dense matrix dense vector multiplication: d_o1 = d_m * d_b
    {
        clock_t t;
        t = clock();
        for(int k=0;k<bench_iterations;k++)  bench_Dense(im2col_mat_tr, filter_vectorized, d_o1);
        double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
        t_im2col+=elapsed/(Ih*Iw*1.0); // normalized timing
    }
    
    // Print out the o1 from im2col:
    std::cout << "\n===im2col Output with Size: " << d_o1.rows() << ", " << d_o1.cols() <<  " \n" << d_o1 << std::endl;
    cout << "-----\n" << endl;
    
    
    // Prepare the Adata, Aindices, AindPtr for CSR multiplication
    int nz = lowered_mat_sparse.nonZeros();
    vector<double> Adata (lowered_mat_sparse.valuePtr(), lowered_mat_sparse.valuePtr() + nz);
    vector<int> Aindices (lowered_mat_sparse.innerIndexPtr(), lowered_mat_sparse.innerIndexPtr() + nz);
    vector<int> Aindptr (lowered_mat_sparse.outerIndexPtr(), lowered_mat_sparse.outerIndexPtr() + lowered_mat_sparse.outerSize()); // +1 for the last element
    // push back the last element the number of nnz in ptr:
    Aindptr.push_back(nz);
    
    
    // Perform 50 times raw sparse matrix dense vector multiplication of CPO: d_o3 = d_m * d_b
    {
        clock_t t;
        t = clock();
        for(int k=0;k<1;k++) CPO(d_o3, filter_vectorized, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
        //      for(int k=0;k<1;k++) _CPO(d_o3, filter_vectorized, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
        double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
        t_csr+=elapsed/(Ih*Iw*1.0); // normalized timing
    }
    
    // Perform 50 times raw sparse matrix dense vector multiplication: d_o2 = d_m * d_b
    //   {
    //      clock_t t;
    //      t = clock();
    //      // for(int k=0;k<bench_iterations;k++) csrMult(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
    //      // for(int k=0;k<1;k++) csrMult_v1(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
    //      for(int k=0;k<1;k++) csrMult_vp(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
    //      double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds 
    //      t_csr+=elapsed/(Ih*Iw*1.0); // normalized timing
    //   } 
    
    // Test the sparse rep of the org im2col
    // Perform 50 times raw sparse matrix dense vector multiplication: d_o2 = d_m * d_b
    // SparseMatrix<float, RowMajor> s_m = im2col_mat_tr.sparseView();
    //   SparseMatrix<float, RowMajor> s_m = im2col_mat.sparseView();
    //   s_m.makeCompressed();
    //   MatrixXf d_o3 = MatrixXf::Zero(Oh, Ow);
    //    
    //   int t_test = 0;
    //   {  
    //      clock_t t;
    //      t = clock(); 
    //      for(int k=0;k<bench_iterations;k++) bench_Sparse(s_m.transpose(), filter_vectorized, d_o3);
    //      double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds 
    //      t_test+=elapsed/(Ih*Iw*1.0); // normalized timing
    //   } 
    //   cout << d_o3 << endl;
    //   cout << "t_test: " << t_test << endl; 
    
    // Print out the o1 from im2col:
    std::cout << "\n===CSCC Output with Size: " << d_o2.rows() << ", " << d_o2.cols() <<  " \n" << d_o2 << std::endl;
    cout << "-----\n" << endl;
    
    // elapsed time per feature element in the entire bench iterations
    std::cout<<"\nbatch\t"<<In<<"\tdensity\t"<<density<<"\tim2col\t"<< t_im2col <<"\tcsr\t"<< t_csr <<std::endl;
    
    // std::cout<<"\nbatch\t"<<In<<"\tdensity\t"<<density<<"\tim2col\t"<<t_im2col/bench_iterations/iter <<"\tcsr\t"<<t_csr/bench_iterations/iter<<std::endl;
    //  csrMult(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
    //  cout << d_o2 ;
    
    /*
     std::cout << "\n===Sparse Feature Map: \n" << sm << std::endl;
     cout << "-----\n" << endl;
     std::cout << " Size of Ptr:  " << sm.outerSize() << endl;
     
     nz = sm.nonZeros();
     std::cout << "non_zeros : " << nz << " density: " << nz/(sm.size()*1.0) << std::endl;
     
     for (auto it = sm.valuePtr(); it != sm.valuePtr() + nz; ++it)
     std::cout << *it << std::endl;
     */
    return 0;
}
