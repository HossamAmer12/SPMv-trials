#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
// #include <boost/timer/timer.hpp>
#include <time.h>
#include <fstream>
// #include <sys/time.h>
#include <omp.h>
#include <math.h>

#define IS_PRINT        0
#define IS_PRINT_SIZE   0
#define PARALLEL        0
#define NUM_THREAD 2
#define MAXTHREAD 16


using namespace Eigen;
using namespace std;
// using namespace boost::timer;

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




void sp_mv_product_vp_old(vector<double>& O, vector<double>& Adata, vector<int>& Aindptr, vector<int>& Aindices, vector<double>& K)
{
    int n = Aindptr.size();
    //! Compute the sparse matrix-vector product (CSR format)  
    int row_ind, index;
    int n_threads = omp_get_num_procs();
    // omp_set_num_threads(NUM_THREAD);
    omp_set_num_threads(n_threads);
    // omp_set_num_threads(Ow);
#pragma omp parallel for default(none) \
  private(row_ind,index) shared(O , Adata, Aindptr, Aindices , K , n)

    for (row_ind = 0; row_ind < n; row_ind++) {
        O[row_ind] = 0.0;
        for (index = Aindptr[row_ind]; index < Aindptr[row_ind + 1]; index++) {
            O[row_ind] += Adata[index] * K[Aindices[index]];
        }
    }
}

typedef float data_t;
typedef int index_t;


void sp_mv_product_vp_v1(vector<double>& O, vector<double>& Adata, vector<int>& Aindptr, vector<int>& Aindices, vector<double>& K)
{
    index_t n = Aindptr.size();
    //! Compute the sparse matrix-vector product (CSR format)  
    index_t row_ind, index;
    // int n_threads = omp_get_num_procs();
    // omp_set_num_threads(NUM_THREAD);
    omp_set_num_threads(4);
    // omp_set_num_threads(Ow);
    // N : number of non-zerp elements 
    index_t N = Aindptr[Aindptr.size() - 1];
#pragma omp parallel for private (index, row_ind) schedule(static, 10)
    for (int i = 0; i < N - 1; i++)
    {
        O[i] = 0.0;
        for (row_ind = 0; row_ind < n; row_ind++) {
            // O[row_ind] = 0.0;
            for (index = Aindptr[row_ind]; index < Aindptr[row_ind + 1]; index++) {
                O[i] += Adata[index] * K[Aindices[index]];
            }
        }
    }
}

void sp_mv_product_vp_v2(vector<double>& O, vector<double>& Adata, vector<int>& Aindptr, vector<int>& Aindices, vector<double>& K)
{
    index_t nnz = Aindptr.size();
    omp_set_num_threads(8);

#pragma omp parallel for
    for (index_t idx = 0; idx < nnz; idx++)
    {
        data_t mval = Adata[idx]; // sparse
        index_t r = Aindptr[idx];
        index_t c = Aindices[idx];
        data_t xval = K[c]; // filter 
        data_t prod = mval * xval;
#pragma omp atomic
        O[r] += prod;
    }
}

void sp_mv_product(vector<double>& O, vector<double>& Adata, vector<int>& Aindptr, vector<int>& Aindices, vector<double>& K)
{
    int n = Aindptr.size();
    //! Compute the sparse matrix-vector product (CSR format)  
    int row_ind, index;

    for (row_ind = 0; row_ind < n; row_ind++) {
        O[row_ind] = 0.0;
        for (index = Aindptr[row_ind]; index < Aindptr[row_ind + 1]; index++) {
            O[row_ind] += Adata[index] * K[Aindices[index]];
        }
    }
}


// double csr_seq( vector<int>& *m, vec_t *x, index_t r)
double csr_seq(vector<int>& Aindptr, vector<int>& Aindices, vector<double>& Adata, vector<double>& K, int r)
{
    index_t idxmin = Aindptr[r];
    index_t idxmax = Aindptr[r + 1];
    data_t val = 0.0;
    for (int idx = idxmin; idx < idxmax; idx++)
    {
        index_t c = Aindices[idx];
        data_t mval = Adata[idx];
        data_t xval = K[c];
        val += mval * xval;
    }
    return val;
}

void sp_mv_product_seq(vector<double>& O, vector<double>& Adata, vector<int>& Aindptr, vector<int>& Aindices, vector<double>& K)
{
    index_t nrow = O.size();
    for (index_t r = 0; r < nrow; r++)
    {
        O[r] = csr_seq(Aindptr, Aindices, Adata, K, r);
    }
}

void sp_mv_product_vp(vector<double>& O, vector<double>& Adata, vector<int>& Aindptr, vector<int>& Aindices, vector<int>& rindex, vector<double>& K)
{
    index_t nnz = Adata.size();
    index_t nrow = O.size();
#pragma omp parallel for
    for (index_t idx = 0; idx < nnz; idx++)
    {
        data_t mval = Adata[idx];
        index_t r = rindex[idx];
        index_t c = Aindices[idx];
        data_t xval = K[c];
        data_t prod = mval * xval;
#pragma omp atomic
        O[r] += prod;
    }
}
// Dense representation of vector
typedef struct {
    int length;
    // Value accessed
    double* value;
} vec_t;

void zero_vector(vec_t* vec) {
    memset(vec->value, 0, vec->length * sizeof(int));
}

vec_t* new_vector(int length)
{
    vec_t* v = (vec_t*)malloc(sizeof(vec_t));
    v->length = length;
    v->value = (double*)calloc(length, sizeof(double));
    return v;
}


void sp_mv_product_vp1(vector<double>& O, vector<double>& Adata, vector<int>& Aindptr, vector<int>& Aindices, vector<int>& rindex, vector<double>& K)
{
    clock_t t;
    index_t nnz = Adata.size();
    index_t nrow = Aindptr.size() - 1;
    index_t n_threads = omp_get_num_procs();
    vec_t* scratch_vector[MAXTHREAD];
    // init_scratch_vectors(nrow, scratch_vector);
    for (index_t t = 0; t < MAXTHREAD; t++)
    {
        scratch_vector[t] = new_vector(nrow);
    }

    omp_set_num_threads(MAXTHREAD);
    // omp_set_num_threads(1);
    // cout <<"rindex : " << rindex.size()<<"cindex : " << Aindices.size() <<endl;
    t = clock();
#pragma omp parallel
    {
        index_t tid = omp_get_thread_num();
        index_t tcount = omp_get_num_threads();
        vec_t* svec = scratch_vector[tid];

        memset(svec->value, 0, svec->length * sizeof(int));
        // zero_vector(svec);

#pragma omp for
        for (int idx = 0; idx < nnz; idx++)
        {
            data_t mval = Adata[idx];
            index_t     r = rindex[idx];
            index_t     c = Aindices[idx];
            data_t xval = K[c];
            data_t prod = mval * xval;
            svec->value[r] += prod;
        }
#pragma omp for
        for (index_t r = 0; r < nrow; r++)
        {
            data_t val = 0.0;
            for (index_t t = 0; t < tcount; t++)
            {
                val += scratch_vector[t]->value[r];
            }
            O[r] = val;
        }
    }
    double elapsed_ = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds 
    cout << "INNER**CSR with omp v2 : " << elapsed_ << " milliseconds" << endl;
}


void sp_mv_product_vp2(vector<double>& O, vector<double>& Adata, vector<int>& Aindptr, vector<int>& Aindices, vector<int>& rindex, vector<double>& K)
{
    clock_t t;
    index_t nnz = Adata.size();
    t = clock();
#pragma omp parallel
    {
        // omp_set_num_threads();
        data_t val = 0.0;
        index_t last_r = 0;
#pragma omp for nowait
        for (index_t idx = 0; idx < nnz; idx++) {
            data_t mval = Adata[idx];
            index_t r = rindex[idx];
            index_t c = Aindices[idx];
            data_t xval = K[c];
            data_t prod = mval * xval;
            if (r == last_r) {
                val += prod;
            }
            else {
#pragma omp atomic
                O[last_r] += val;
                last_r = r;
                val = prod;
            }
        }
#pragma omp atomic
        O[last_r] += val;
    }
    double elapsed_ = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds 
    cout << "INNER**CSR with omp v3 : " << elapsed_ << " milliseconds" << endl;
}




void print_vec(vector<double>& Op)
{
    for (int i = 0; i < Op.size(); i++)
    {
        std::cout << Op.at(i) << ' ';
    }
    std::cout << std::endl;
}


void print_vec(vector<int>& Op)
{
    for (int i = 0; i < Op.size(); i++)
    {
        std::cout << Op.at(i) << ' ';
    }
    std::cout << std::endl;
}


// void eign2vector2D (MatrixXf& m, vector<double>& Adata)
// {
//   std::vector<std::vector<double>> v;
//   for (int i=0; i<m.rows(); ++i)
//   {
//       // const float* begin = &m.row(i).data()[0];
//     const float* begin = m.col(i).data();
//       Adata.push_back(std::vector<float>(begin, begin+m.cols()));
//   }
// }

int main()
{

    // float density:
    float density = 0.05;
    // float density = 0.5;
    for (; density < 1.05; density += 0.05)
    {
        cout << " ******* DENSITY =  " << density << " ******* " << endl;

        // timer for im2col, csr
        float t_im2col = 0;
        float t_csr = 0;
        float t_csr_vp = 0;
        float t_csr_vp1 = 0;


        // bench iterations
        int bench_iterations = 1000;


        // Conv parameters:
        int padding = 0;
        int stride = 1;
        int num_filters = 1; // 64

        // mixed 0: conv_node 3
      //  int Ih = 5;
      //  int Iw = 5;

        //int Ih = 149;
        //int Iw = 149;

         int Ih = 20;
         int Iw = 20;

        // Kernel dimensions
        int Kh = 3;
        int Kw = 3;


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

        std::vector<int> cols = { 0,1,4,0,4,0,4 };
        std::vector<int> rows = { 0,0,0,2,2,3,3 };
        std::vector<double> values = { 1,1,1,1,1,1,1 };



        for (int i = 0; i < ceil(density * Ih * Iw); ++i)
        {
            int r = rand() % Ih;
            int c = rand() % Iw;
            org_fm(r, c) = 1;
        }

#if IS_PRINT
        // Print out the original feature map:
        std::cout << "\n===Original Feature Map (" << Ih << "x" << Iw << "):  \n" << org_fm << std::endl;
        cout << "-----\n" << endl;
#endif


#if IS_PRINT_SIZE
        // Print out the original feature map:
        std::cout << "\n===Original Feature Map (" << Ih << "x" << Iw << "):  \n" << std::endl;
        cout << "-----\n" << endl;
#endif  

        // Create the lowered matrix: 
        MatrixXf lowered_mat = MatrixXf::Zero(Ow, Ih * Kw);
        int sub_matrix_index = 0;

        //std::cout << "\n====================================================================================\n" << endl;
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

#if IS_PRINT  
  // Print out the lowered feature map:
        std::cout << "\n===Lowered Feature Map of Size: " << lowered_mat.rows() << ", " << lowered_mat.cols() << "\n" << lowered_mat << std::endl;
        cout << "-----\n" << endl;
#endif


#if IS_PRINT_SIZE  
        // Print out the lowered feature map:
        std::cout << "\n===Lowered Feature Map of Size: " << lowered_mat.rows() << ", " << lowered_mat.cols() << "\n" << std::endl;
        cout << "-----\n" << endl;
#endif

        int start_row_int = 0;
        int start_col_int = 0;

        // Create the intermediate representation for im2col:
        MatrixXf im2col_mat = MatrixXf::Zero(Oh * Ow, Kh * Kw * Ic);
        int patch_matrix_index = 0;

        // For each patch:
        for (int patch = 0; patch < Oh * Ow; patch = patch + stride)
        {

            int im2col_mat_col_index = 0;

            // Fetch this piece (all rows, all cols in this current submatrix)
            for (int row_int = start_row_int; row_int < start_row_int + Kh; ++row_int)
            {
                for (int col_int = start_col_int; col_int < start_col_int + Kw; ++col_int)
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

            if (start_row_int + Kh > Ih)
            {
                start_row_int = 0;
                start_col_int = start_col_int + stride;
            }


            if (start_col_int + Kw > Iw)
            {
                break;
            }

        } // end outer outer loop

#if IS_PRINT
  // Print out the im2col interedmiate feature map:
        std::cout << "\n===im2col Intermediate Feature Map with Size: " << im2col_mat.rows() << ", " << im2col_mat.cols() << " \n" << im2col_mat << std::endl;
        cout << "-----\n" << endl;
#endif


#if IS_PRINT_SIZE
        // Print out the im2col interedmiate feature map:
        std::cout << "\n===im2col Intermediate Feature Map with Size: " << im2col_mat.rows() << ", " << im2col_mat.cols() << " \n" << std::endl;
        cout << "-----\n" << endl;
#endif

        // Create the sparse representation of the lowered matrix:
        SparseMatrix<float, RowMajor> lowered_mat_sparse = lowered_mat.sparseView();
        // std::vector<Eigen::Triplet<bool> > lowered_mat_sparse = lowered_mat.sparseView();;
        // SparseMatrix<int> lowered_mat_sparse = lowered_mat.sparseView();
        lowered_mat_sparse.makeCompressed();

#if IS_PRINT 
        // Print out the im2col interedmiate feature map:
        std::cout << "\n===CSR of Lowered Feature Map: " << " \n" << lowered_mat_sparse << std::endl;
        cout << "-----\n" << endl;
#endif  

        // Create the filter K and its vectorized version:
        MatrixXf filter = MatrixXf::Ones(Kh, Kw);
        VectorXf filter_vectorized = VectorXf::Ones(Kh * Kw);


#if IS_PRINT
        // Print out the im2col interedmiate feature map:
        std::cout << "\n===Filter: " << " \n" << filter_vectorized << std::endl;
        cout << "-----\n" << endl;
#endif

#if IS_PRINT_SIZE
        // Print out the im2col interedmiate feature map:
        std::cout << "\n===Filter: " << " \n" << filter_vectorized << std::endl;
        cout << "-----\n" << endl;
#endif

        // Prepare the output for im2col, sparseMat
        MatrixXf d_o1 = MatrixXf::Zero(Oh, Ow);
        MatrixXf d_o2 = MatrixXf::Zero(Oh, Ow);

        // transpose the matrix for im2col:
        MatrixXf im2col_mat_tr = im2col_mat.transpose();

        // Perform 50 times dense matrix dense vector multiplication: d_o1 = d_m * d_b
        {
            clock_t t;
            t = clock();
            for (int k = 0; k < bench_iterations; k++)  bench_Dense(im2col_mat, filter_vectorized, d_o1);
            double elapsed = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds 
            // double elapsed =  omp_get_wtime() - t;
            t_im2col += elapsed / (Ih * Iw * 1.0); // normalized timing
        }

#if IS_PRINT
        // Print out the o1 from im2col:
        std::cout << "\n===im2col Output with Size: " << d_o1.rows() << ", " << d_o1.cols() << " \n" << d_o1 << std::endl;
        cout << "-----\n" << endl;
#endif


#if IS_PRINT_SIZE 
        // Print out the o1 from im2col:
        std::cout << "\n===im2col Output with Size: " << d_o1.rows() << ", " << d_o1.cols() << std::endl;
        cout << "-----\n" << endl;
#endif

        // Prepare the Adata, Aindices, AindPtr for CSR multiplication
        int nz = lowered_mat_sparse.nonZeros();
        vector<double> Adata(lowered_mat_sparse.valuePtr(), lowered_mat_sparse.valuePtr() + nz);
        vector<int> Aindices(lowered_mat_sparse.innerIndexPtr(), lowered_mat_sparse.innerIndexPtr() + nz);
        vector<int> Aindptr(lowered_mat_sparse.outerIndexPtr(), lowered_mat_sparse.outerIndexPtr() + lowered_mat_sparse.outerSize()); // +1 for the last element
        // push back the last element the number of nnz in ptr:
        Aindptr.push_back(nz);

        vector<int> rindex;
        int row_start = 1;
        for (std::size_t i = 0; i < Aindptr.size() - 1; i++)
        {
            //cout << "Current --> " << row_start << "\n";
            for (std::size_t row_i = Aindptr[i]; row_i < Aindptr[i + 1]; row_i++)
            {
                rindex.push_back(row_start);
            }
            row_start++;
        }

        vector<int> r_index = rindex;
        vector<int> c_index = Aindices;
        vector<double> cooValue = Adata;

        ofstream myfile_coo;
        char buffer[200];
        int n;
        #ifdef _WIN32
        // cout << "_WIN32" << endl;
        n = sprintf_s(buffer, "D:\\9.CPO&CPS\\OUTPUTs\\loop_conv\\COO\\coo_%d_%0.2f.mtx", Iw, density);
        #endif
        #ifdef linux
        // cout << "linux" << endl;
        n = sprintf(buffer, "/home/ahamsala/scratch/DC/openmp_projects/merge-spmv/tgz/coo_105/coo_%d_%0.2f.mtx", Iw, density);
        #endif
        myfile_coo.open(buffer, ios::out);
        myfile_coo << lowered_mat.rows() << " " << lowered_mat.cols() << " " << nz << endl;
        for (int i = 0; i < nz; i++)
        {
            myfile_coo << r_index[i] << " " << c_index[i] << endl;
        }
        myfile_coo.close();


        double t;
        clock_t t_;
        double elapsed_, elapsed;
#if IS_PRINT

        std::cout << "\n===Lowered Feature Map of Size: " << lowered_mat.rows() << ", " << lowered_mat.cols() << "\n" << lowered_mat << std::endl;

        cout << "Adata Vectors : " << endl;
        print_vec(Adata);

        cout << "Aindices Vectors : " << endl;
        print_vec(Aindices);

        cout << "Aindptr Vectors : " << endl;
        print_vec(Aindptr);
#endif
#if IS_PRINT
        // vector<int> rindex;
        // int row_start = 0;
        // for (int i; i < Aindptr.size()-1 ; i++ )
        // {
        //   // cout << "start --> " << Aindptr[i] << " end --> " <<Aindptr[i+1]<<endl;
        //   for (int row_i = Aindptr[i]; row_i < Aindptr[i+1]; row_i++) 
        //     rindex.push_back(row_start);
        //   row_start++;
        // }
        // print_vec(rindex);
        vector<double> Op(lowered_mat.rows(), 0.0);
        vector<double> K1(lowered_mat.cols(), 1.0);
        t = clock();
        sp_mv_product(Op, Adata, Aindptr, Aindices, K1);
        elapsed_ = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds 
        cout << "*CSR without omp : " << elapsed_ << " milliseconds" << endl;
        print_vec(Op);
        std::fill(Op.begin(), Op.end(), 0);
        t = clock();
        sp_mv_product_seq(Op, Adata, Aindptr, Aindices, K1);
        elapsed_ = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds 
        cout << "*CSR with element by element without omp: " << elapsed_ << " milliseconds" << endl;
        print_vec(Op);
        // version 1
        std::fill(Op.begin(), Op.end(), 0);
        t = clock();
        sp_mv_product_vp(Op, Adata, Aindptr, Aindices, rindex, K1);
        elapsed_ = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds 
        cout << "*CSR with omp : " << elapsed_ << " milliseconds" << endl;
        print_vec(Op);

        // Version 2
        std::fill(Op.begin(), Op.end(), 0);
        t = clock();
        sp_mv_product_vp1(Op, Adata, Aindptr, Aindices, rindex, K1);
        elapsed_ = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds 
        cout << "*CSR with omp v2 : " << elapsed_ << " milliseconds" << endl;
        print_vec(Op);

        // version 3
        std::fill(Op.begin(), Op.end(), 0);
        t = clock();
        sp_mv_product_vp2(Op, Adata, Aindptr, Aindices, rindex, K1);
        elapsed_ = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in milliseconds 
        cout << "*CSR with omp v3 : " << elapsed_ << " milliseconds" << endl;
        print_vec(Op);

        return 0;
        // ofstream myfile;
        // myfile.open ("csr_log.txt", ios::out | ios::app);
        // int batch = 1;
        // myfile << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batch\t"<<batch<<"\tdensity\t"<<density<<"\tim2col\t"<<t_im2col<<"\tcsr\t"
        // <<t_csr <<"\tpercent\t"<< 100.0*(t_im2col-t_csr)/t_im2col << "\n";
        // myfile.close();

        // printf("******************************************************\n");
        // double t1 , elapsed1;
        // t1 = omp_get_wtime();
        // sp_mv_product(Op, Adata, Aindptr, Aindices, K1);
        // elapsed1  = omp_get_wtime()-t1;
        // cout << "CSR without omp : " << elapsed1 << " seconds"<< endl;
        // print_vec(Op);
        // std::fill(Op.begin(), Op.end(), 0);

        // t1 = omp_get_wtime();
        // sp_mv_product_vp(Op, Adata, Aindptr, Aindices, K1);
        // elapsed1  = omp_get_wtime()-t1;
        // cout << "CSR with omp : " << elapsed1 << " seconds"<< endl;
        // print_vec(Op);
        // std::fill(Op.begin(), Op.end(), 0);

        cout << " sp_mv_product_vp " << endl;
        cout << "Aindptr : " << Aindptr.size()
            << " Aindices : " << Aindices.size()
            << " Adata : " << Adata.size()
            << " output: " << Op.size()
            << " filter : " << K1.size()
            << endl;

#endif 

        // return 0;

        // #if PARALLEL

        //    // Perform 50 times raw sparse matrix dense vector multiplication: d_o2 = d_m * d_b
        //    {  
        //       struct timeval tv1, tv2;
        //       struct timezone tz;

        //       cout << " *** comp scope *** \n" <<endl;

        //       // cout << " csrMult v2" <<endl;
        //       cout << " csrMult " <<endl;
        //       // gettimeofday(&tv1, &tz);
        //       t_ = clock();
        //       t = omp_get_wtime(); 
        //       // for(int k=0;k<bench_iterations;k++) csrMult(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
        //       for(int k=0;k<bench_iterations;k++) csrMult(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
        //       elapsed = omp_get_wtime()-t;
        //       elapsed_ = 1000*((double)(clock()-t_))/CLOCKS_PER_SEC; // time in milliseconds 
        //       // gettimeofday(&tv2, &tz);
        //       cout << "normal csr  --> (omp clock )elapsed time = " << elapsed << " seconds \t"  
        //            << " elapsed time = " << elapsed_ << " milliseconds"<< endl; 
        //       d_o2 = d_o2 / bench_iterations;
        //       t_csr+=elapsed_/(Ih*Iw*1.0); // normalized timing`
        //  #if IS_PRINT
        //   // Print out the o1 from im2col:
        //   std::cout << "\n===CSCC Output with Size: " << d_o2.rows() << ", " << d_o2.cols() <<  " \n" << d_o2 << std::endl;
        //   cout << "-----\n" << endl;
        // #endif     
        //       // elapsed_ = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
        //       // double elapsed = 1000*((double)(omp_get_wtime()-t))/CLOCKS_PER_SEC; // time in milliseconds 

        //       // -------------------------------- Second approach --------------------------
        //     }
        //     {
        //       cout << "\n csrMult_vp \n" <<endl;
        //       MatrixXf d_o2 = MatrixXf::Zero(Oh, Ow);
        //       t_ = clock();
        //       t = omp_get_wtime(); 
        //       for(int k=0;k<bench_iterations;k++) csrMult_vp(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
        //       elapsed = omp_get_wtime()-t;
        //       elapsed_ = 1000*((double)(clock()-t_))/CLOCKS_PER_SEC; // time in milliseconds
        //       cout << "Parallel csr  --> (omp clock )elapsed time = " << elapsed << " seconds \t"  
        //            << " elapsed time = " << elapsed_  << " milliseconds"<< endl; 
        //       d_o2 = d_o2 / bench_iterations;
        //       t_csr_vp+=elapsed_/(Ih*Iw*1.0);
        // #if IS_PRINT
        //   // Print out the o1 from im2col:
        //   std::cout << "\n===CSCC Output with Size: " << d_o2.rows() << ", " << d_o2.cols() <<  " \n" << d_o2 << std::endl;
        //   cout << "-----\n" << endl;
        // #endif
        //     }
        //     {
        //       cout << "\n csrMult_vp1 \n" <<endl;
        //       MatrixXf d_o2 = MatrixXf::Zero(Oh, Ow);
        //       t_ = clock();
        //       t = omp_get_wtime(); 
        //       for(int k=0;k<bench_iterations;k++) csrMult_vp1(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
        //       elapsed = omp_get_wtime()-t;
        //       elapsed_ = 1000*((double)(clock()-t_))/CLOCKS_PER_SEC; // time in milliseconds
        //       cout << "Parallel csr  --> (omp clock )elapsed time = " << elapsed << " seconds \t"  
        //            << " elapsed time = " << elapsed_  << " milliseconds"<< endl; 
        //       d_o2 = d_o2 / bench_iterations;
        //       t_csr_vp1+=elapsed_/(Ih*Iw*1.0);
        // #if IS_PRINT
        //   // Print out the o1 from im2col:
        //   std::cout << "\n===CSCC Output with Size: " << d_o2.rows() << ", " << d_o2.cols() <<  " \n" << d_o2 << std::endl;
        //   cout << "-----\n" << endl;
        // #endif

        //       // for(int k=0;k<bench_iterations;k++) csrMult_v1(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
        //       // bench_iterations = 1; // if you want to see the correct result of csr_mult, comment this line
        //       // for(int k=0;k<bench_iterations;k++) csrMult_v2(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);

        //    }

        //    // cout << " do2 :" << d_o2 << "\n do1 :" << d_o1 << endl;
        //    // MatrixXf sub = d_o2 - d_o2;
        //    // cout << "Subtraction : " <<  d_o2 - d_o1 <<endl;


        //   // elapsed time per feature element in the entire bench iterations
        //   std::cout<<"batch = "<<In<<" --> density = "<<density<<" // im2col --> "<< t_im2col 
        //   <<"\ncsr --> "<< t_csr << " // csr_vp --> " << t_csr_vp << " // csr_vp1 --> " << t_csr_vp1
        //   <<"\npercent_csr --> "<< 100.0*(t_im2col-t_csr)/t_im2col 
        //   <<" // percent_vp --> "<< 100.0*(t_im2col-t_csr_vp)/t_im2col 
        //   <<" // percent_vp1 --> "<< 100.0*(t_im2col-t_csr_vp1)/t_im2col 
        //   <<std::endl;
        // #endif

        ofstream myfile;
        #ifdef _WIN32
        n = sprintf_s(buffer, "D:\\9.CPO&CPS\\OUTPUTs\\loop_conv\\csr_log.txt", nz, density);
        #endif
        #ifdef linux
        n = sprintf(buffer, "/home/ahamsala/scratch/DC/openmp_projects/merge-spmv/tgz/coo_105/coo_%d_%0.2f.mtx", nz, density);
        #endif
        myfile.open("D:\\9.CPO&CPS\\OUTPUTs\\loop_conv\\csr_log.txt", ios::out | ios::app);
        int batch = 1;
        myfile << Kh << "x" << Kw << " | " << Ih << "x" << Iw << ") batch\t" << batch << "\tdensity\t" << density << "\tim2col\t" << t_im2col << "\tcsr\t"
            << t_csr << "\tpercent\t" << 100.0 * (t_im2col - t_csr) / t_im2col << "\n";
        myfile.close();

    } // density loop
    return 0;
}
