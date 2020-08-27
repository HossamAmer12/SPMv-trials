#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <time.h>
#include <fstream>
#include <omp.h>
#include <math.h>
#include <string>


#define IS_PRINT        0
#define IS_PRINT_SIZE   0
#define gen_mtx         1
#define NUM_THREAD_SPMV 4
#define NUM_THREAD_CONV 4
#define MAXTHREAD       16
#define SpMV_seq_para   1
#define Conv_CSR        0


using namespace Eigen;
using namespace std;
// using namespace boost::timer;

// https://scicomp.stackexchange.com/questions/27977/how-can-i-speed-up-this-code-for-sparse-matrix-vector-multiplication


// Dense representation of vector


typedef double data_t;
typedef int index_t;

typedef struct {
    int length;
    // Value accessed
    data_t* value;
} vec_t;

typedef struct {
    int length;
    // Value accessed
    index_t* value;
} vec_t_i;

void print_vec(vector<data_t>& Op);

void print_vec(vector<index_t>& Op);

void print_vec(vec_t* vec);

void free_vector(vec_t* vec);

void free_vector(vec_t_i* vec);



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


void csrMult_vp(MatrixXf& O, VectorXf& K, vector<double>& Adata, vector<int>& Aindices, vector<int>& Aindptr, int Kh, int Kw, int Oh, int Ow)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    int x = Aindptr[0];
    int* Aindex_help = &Aindices[x];
    double* Adata_help = &Adata[x];
    int n, l;
    omp_set_num_threads(NUM_THREAD_CONV);
    // #pragma omp parallel for private (Aindex_help, Adata_help) shared(O , n)
    for (n = 0; n < Ow; ++n)
    {
        for (; x < Aindptr[n + 1]; ++x)
        {
            double result = 0.0;
            int NZE_index = *Aindex_help; Aindex_help++;
            int NZE_data = *Adata_help; Adata_help++;
            for (l = 0; l < Kh; ++l)
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


void csrMult_vp1(MatrixXf& O, VectorXf& K, vector<double>& Adata, vector<int>& Aindices, vector<int>& Aindptr, int Kh, int Kw, int Oh, int Ow)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    int x = Aindptr[0];
    int* Aindex_help = &Aindices[x];
    double* Adata_help = &Adata[x];
    int n, l;
    omp_set_num_threads(NUM_THREAD_CONV);
    //#pragma omp parallel for private (n, x, l)
    for (n = 0; n < Ow; ++n)
    {
        for (; x < Aindptr[n + 1]; ++x)
        {
            double result = 0.0;
            int NZE_index = *Aindex_help; Aindex_help++;
            int NZE_data = *Adata_help; Adata_help++;
            for (l = 0; l < Kh; ++l)
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



void sp_mv_product_vp_v1(vector<double>& O, vector<double>& Adata, vector<int>& Aindptr, vector<int>& Aindices, vector<double>& K)
{
    index_t n = Aindptr.size();
    //! Compute the sparse matrix-vector product (CSR format)  
    index_t row_ind, index;
    // int n_threads = omp_get_num_procs();
    // omp_set_num_threads(NUM_THREAD);
    omp_set_num_threads(NUM_THREAD_SPMV);
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
    omp_set_num_threads(NUM_THREAD_SPMV);

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

void sp_mv_product(vector<data_t>& O, vector<data_t>& Adata, vector<index_t>& Aindptr, vector<index_t>& Aindices, vector<data_t>& K)
{
    int n = O.size();
    //! Compute the sparse matrix-vector product (CSR format)  
    size_t row_ind, index;

    for (row_ind = 0; row_ind < n; row_ind++) {
        O[row_ind] = 0.0;
        for (index = Aindptr[row_ind]; index < Aindptr[row_ind + 1]; index++)
        {
            O[row_ind] += Adata[index] * K[Aindices[index]];
        }
    }
}


// double csr_seq( vector<int>& *m, vec_t *x, index_t r)
double csr_seq(vector<index_t>& Aindptr, vector<index_t>& Aindices, vector<data_t>& Adata, vector<data_t>& K, int r)
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

void sp_mv_product_seq(vector<data_t>& O, vector<data_t>& Adata, vector<index_t>& Aindptr, vector<index_t>& Aindices, vector<data_t>& K)
{
    index_t nrow = O.size();
    for (index_t r = 0; r < nrow; r++)
    {
        O[r] = csr_seq(Aindptr, Aindices, Adata, K, r);
    }
}

typedef struct {
    index_t nrow;       // Number of rows (= number of columns)
    index_t nnz;        // Number of nonzero elements
    data_t* value;      // Nonzero matrix values, row-major order [nnz]
    index_t* cindex;    // Column index for each nonzero entry    [nnz]
    index_t* rowstart;  // Offset of each row                     [nrow+1]
    // Following only needed for data-oriented parallelism
    index_t* rindex;     // Row index for each nonzero entry       [nnz]
} csr_t;

csr_t* new_csr(index_t nrow, index_t nnz) {
    csr_t* m = (csr_t*)malloc(sizeof(csr_t));
    m->nrow = nrow;
    m->nnz = nnz;
    m->value = (data_t*)calloc(nnz, sizeof(data_t));
    m->cindex = (index_t*)calloc(nnz, sizeof(index_t));
    m->rowstart = (index_t*)calloc(nrow + 1, sizeof(index_t));
    m->rindex = (index_t*)calloc(nnz, sizeof(index_t));
    return m;
}

void free_csr(csr_t* m) {
    free(m->value);
    free(m->cindex);
    free(m->rowstart);
    free(m->rindex);
    free(m);
}

void zero_vector(vec_t* vec) {
    memset(vec->value, 0, vec->length * sizeof(index_t));
}

vec_t* new_vector(int length)
{
    vec_t* v = (vec_t*)malloc(sizeof(vec_t));
    v->length = length;
    v->value = (data_t*)calloc(length, sizeof(data_t));
    return v;
}

vec_t_i* new_vector_i(int length)
{
    vec_t_i* v = (vec_t_i*)malloc(sizeof(vec_t_i));
    v->length = length;
    v->value = (index_t*)calloc(length, sizeof(index_t));
    return v;
}

void free_vector(vec_t* vec) {
    free(vec->value);
    free(vec);
    // cout << "vec_t" << endl;
}

void free_vector(vec_t_i* vec) {
    free(vec->value);
    free(vec);
}

void copy_vec(vec_t* to, vector<data_t>& from)
{
    for (int i = 0; i < from.size(); i++)
    {
        to->value[i] += from[i];
    }
}

void copy_vec(vec_t_i* to, vector<index_t>& from)
{
    for (int i = 0; i < from.size(); i++)
    {
        to->value[i] += from[i];
    }
}

void sp_mv_product_vp(vec_t* O, vector<data_t>& Adata, vector<index_t>& Aindptr, vector<index_t>& Aindices, vector<index_t>& rindex, vector<data_t>& K, index_t nrow)
{
    index_t nnz = Adata.size();
    // index_t nrow = O.size();
    zero_vector(O);
#pragma omp parallel for
    for (index_t idx = 0; idx < nnz; idx++)
    {
        data_t mval = Adata[idx];
        index_t r = rindex[idx];
        index_t c = Aindices[idx];
        data_t xval = K[c];
        data_t prod = mval * xval;
#pragma omp atomic
        O->value[r] += prod;
    }
}

void sp_mv_product_vp_(vec_t* O, vec_t* Adata, vec_t_i* Aindptr, vec_t_i* Aindices, vec_t_i* rindex, vec_t* K, index_t nrow, index_t nnz)
{
    // index_t nrow = O.size();
    zero_vector(O);
#pragma omp parallel for
    for (index_t idx = 0; idx < nnz; idx++)
    {
        data_t mval = Adata->value[idx];
        index_t r = rindex->value[idx];
        index_t c = Aindices->value[idx];
        data_t xval = K->value[c];
        data_t prod = mval * xval;
#pragma omp atomic
        O->value[r] += prod;
    }
}

void sp_mv_product_vp1(vector<data_t>& O, vector<data_t>& Adata, vector<index_t>& Aindptr, vector<index_t>& Aindices, vector<index_t>& rindex, vector<data_t>& K)
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

        memset(svec->value, 0, svec->length * sizeof(index_t));
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
    double elapsed_ = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in msec 
    //cout << "INNER**CSR with omp v2 : " << elapsed_ << " msec" << endl;
}


void sp_mv_product_vp1_(vec_t* O, vec_t* Adata, vec_t_i* Aindptr, vec_t_i* Aindices, vec_t_i* rindex, vec_t* K, index_t nrow, index_t nnz)
{
    clock_t t;
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

        memset(svec->value, 0, svec->length * sizeof(index_t));
        // zero_vector(svec);

#pragma omp for
        for (int idx = 0; idx < nnz; idx++)
        {
            data_t mval = Adata->value[idx];
            index_t     r = rindex->value[idx];
            index_t     c = Aindices->value[idx];
            data_t xval = K->value[c];
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
            O->value[r] = val;
        }
    }
    double elapsed_ = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in msec 
    //cout << "INNER**CSR with omp v2 : " << elapsed_ << " msec" << endl;
}


void sp_mv_product_vp2(vector<data_t>& O, vector<data_t>& Adata, vector<index_t>& Aindptr, vector<index_t>& Aindices, vector<index_t>& rindex, vector<data_t>& K)
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
    double elapsed_ = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in msec 
    //cout << "INNER**CSR with omp v3 : " << elapsed_ << " msec" << endl;
}

void sp_mv_product_vp2_(vec_t* O, vec_t* Adata, vec_t_i* Aindptr, vec_t_i* Aindices, vec_t_i* rindex, vec_t* K, index_t nrow, index_t nnz)
{
    //clock_t t;
    zero_vector(O);
    //t = clock();
#pragma omp parallel
    {
        data_t val = 0.0;
        index_t last_r = 0;
        #pragma omp for nowait
        for (index_t idx = 0; idx < nnz; idx++)
        {
            data_t mval = Adata->value[idx];
            index_t r = rindex->value[idx];
            index_t c = Aindices->value[idx];
            data_t xval = K->value[c];
            data_t prod = mval * xval;
            if (r == last_r) {
                val += prod;
            }
            else
            {
                #pragma omp atomic
                O->value[last_r] += val;
                last_r = r;
                val = prod;
            }
        }
        #pragma omp atomic
        O->value[last_r] += val;
    }
    // double elapsed_ = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in msec 
    //cout << "INNER**CSR with omp v3 : " << elapsed_ << " msec" << endl;
}


void print_vec(vector<data_t>& Op)
{
    for (int i = 0; i < Op.size(); i++)
    {
        std::cout << Op.at(i) << ' ';
    }
    std::cout << std::endl;
}


void print_vec(vector<index_t>& Op)
{
    for (int i = 0; i < Op.size(); i++)
    {
        std::cout << Op.at(i) << ' ';
    }
    std::cout << std::endl;
}

void print_vec(vec_t* vec) {
    index_t i;
    printf("[");
    for (i = 0; i < vec->length; i++) {
        printf("\t%.2f", vec->value[i]);
    }
    printf("]\n");
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

int main(int argc, char* argv[])
{
    // Check the number of parameters
    if (argc < 7) {
        // Tell the user how to run the program
        std::cerr << "Usage: " << argv[0] << " density" << " Ih" << " Iw" << " Kh" << " Kw" << " bench_iterations" << std::endl;
        /* "Usage messages" are a conventional way of telling the user
         * how to run a program if they enter the command incorrectly.
         */
        return 1;
    }
    // Print the user's name:
    std::cout << argv[0] << " density : " << argv[1] << " Ih : " << argv[2] << " Iw : " << argv[3]
        << " Kh : " << argv[4] << " Kw : " << argv[5] << " bench_iterations : " << argv[6] << std::endl;

    //Example output(no arguments passed) :
    //Usage: a.exe <NAME>
    //Example output(one argument passed) :
    //a.exe says hello, Chris!

    // float density:
    float density = atof(argv[1]);
    int Ih = atoi(argv[2]);
    int Iw = atoi(argv[3]);
    int Kh = atoi(argv[4]);
    int Kw = atoi(argv[5]);
    int bench_iterations = atoi(argv[6]);
    // bench iterations
    //int bench_iterations = 100;


    // Conv parameters:
    int padding = 0;
    int stride = 1;
    int num_filters = 1; // 64

    // mixed 0: conv_node 3
  //  int Ih = 5;
  //  int Iw = 5;

    //int Ih = 149;
    //int Iw = 149;

    //int Ih = 20;
    //int Iw = 20;
    int Ib = 1; // batch size
    int batch = 1;

    // Kernel dimensions
    //int Kh = 3;
    //int Kw = 3;


    // adjust the iterations based on Ih  
    // if (Ih > 100)
    // {
    //     bench_iterations = 1000;
    // }

    // int Ic = 32; // this is for the node
    int Ic = 1; // put it as articial for now
    int In = 1;

    int K = 1; // number of filters

    int Oh = (1 + Ih - Kh + 2 * padding) / stride; // removed + 1
    int Ow = (1 + Iw - Kw + 2 * padding) / stride;

    int iter = 1;  // total number of times to perform the test for each of dense, sparse multiplication


    // float density = 0.05;

    // for (; density < 1.05; density += 0.05)
    {
        // Create your original input feature map:
        MatrixXf org_fm = MatrixXf::Zero(Ih, Iw);
        cout << " ******* DENSITY =  " << density << " ******* " << endl;

        // timer for im2col, csr
        float t_im2col = 0;
        float t_csr = 0;
        float t_csr_vp = 0;
        float t_csr_vp1 = 0;

        float t_csr_spmv_product = 0;
        float t_csr_spmv_seq = 0;
        float t_csr_spmv_vp = 0;
        float t_csr_spmv_vp1_1 = 0, t_csr_spmv_vp1_2 = 0;
        float t_csr_spmv_vp2_1 = 0, t_csr_spmv_vp2_2 = 0;


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
        // MatrixXf lowered_mat = MatrixXf::Zero(Ow, Ih * Kw);
        MatrixXf lowered_mat = MatrixXf::Zero(1, Ih*Kw);
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
            break;
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
            double elapsed = 1000 * ((double)(clock() - t)) / CLOCKS_PER_SEC; // time in msec 
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
        vector<data_t> Adata(lowered_mat_sparse.valuePtr(), lowered_mat_sparse.valuePtr() + nz);
        vector<index_t> Aindices(lowered_mat_sparse.innerIndexPtr(), lowered_mat_sparse.innerIndexPtr() + nz);
        vector<index_t> Aindptr(lowered_mat_sparse.outerIndexPtr(), lowered_mat_sparse.outerIndexPtr() + lowered_mat_sparse.outerSize()); // +1 for the last element
        // push back the last element the number of nnz in ptr:
        Aindptr.push_back(nz);

        vector<index_t> rindex_;
        {
            int row_start = 1;
            for (std::size_t i = 0; i < Aindptr.size() - 1; i++)
            {
                //cout << "Current --> " << row_start << "\n";
                for (std::size_t row_i = Aindptr[i]; row_i < Aindptr[i + 1]; row_i++)
                {
                    rindex_.push_back(row_start);
                }
                row_start++;
            }
        }

        vector<index_t> r_index = rindex_;
        vector<index_t> c_index = Aindices;
        vector<data_t> cooValue = Adata;
#if gen_mtx
        ofstream myfile_coo;
        char buffer[200];
        int n;
#ifdef _WIN32
         cout << "_WIN32" << endl;
        n = sprintf_s(buffer, "D:\\### PhD Codes ###\\OUTPUTs\\COO\\coo_%d_%d_%d_%0.2f.mtx", Ih, Kh, Kw, density);
#endif
#ifdef linux
        // cout << "linux" << endl;
        n = sprintf(buffer, "/scratch/ahamsala/DC_CODES/OUTPUTs/COO/coo_%d_%d_%d_%0.2f.mtx", Ih, Kh, Kw, density);
#endif
        myfile_coo.open(buffer, ios::out);
        myfile_coo << lowered_mat.rows() << " " << lowered_mat.cols() << " " << nz << endl;
        for (int i = 0; i < nz; i++)
        {
            myfile_coo << r_index[i] << " " << c_index[i] << endl;
        }
        myfile_coo.close();
#endif

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


#if SpMV_seq_para
        vector<index_t> rindex;
        {
            index_t row_start = 0;
            for (int i; i < Aindptr.size() - 1; i++)
            {
                // cout << "start --> " << Aindptr[i] << " end --> " <<Aindptr[i+1]<<endl;
                for (int row_i = Aindptr[i]; row_i < Aindptr[i + 1]; row_i++)
                    rindex.push_back(row_start);
                row_start++;
            }
        }
        vector<data_t> Op(lowered_mat.rows(), 0.0);
        vector<data_t> K1(lowered_mat.cols(), 1.0);
        double bench_iterations_ = bench_iterations;

        // CSR without omp
        t = clock();
        for (int k = 0; k < bench_iterations; k++) sp_mv_product(Op, Adata, Aindptr, Aindices, K1);
        // sp_mv_product(Op, Adata, Aindptr, Aindices, K1);
        elapsed_ = 1000 * ((double)(clock() - t)) / (CLOCKS_PER_SEC * bench_iterations_); // time in msec  
        cout << "CSR without omp : " << elapsed_ << " msec" << endl;
        //print_vec(Op);
        t_csr_spmv_product += elapsed_;


        // CSR with element by element without omp
        std::fill(Op.begin(), Op.end(), 0);
        t = clock();
        for (int k = 0; k < bench_iterations; k++) sp_mv_product_seq(Op, Adata, Aindptr, Aindices, K1);
        // sp_mv_product_seq(Op, Adata, Aindptr, Aindices, K1);
        elapsed_ = 1000 * ((double)(clock() - t)) / (CLOCKS_PER_SEC * bench_iterations_); // time in msec  
        cout << "CSR with element by element without omp: " << elapsed_ << " msec" << endl;
        // //print_vec(Op);
        t_csr_spmv_seq += elapsed_;


        // version 1
        std::fill(Op.begin(), Op.end(), 0);

        index_t nrow = Op.size();
        vec_t* Op_ = new_vector(nrow);

        vec_t* Adata_ = new_vector(Adata.size());
        copy_vec(Adata_, Adata);

        vec_t_i* Aindptr_ = new_vector_i(Aindptr.size());
        copy_vec(Aindptr_, Aindptr);

        vec_t_i* Aindices_ = new_vector_i(Aindices.size());
        copy_vec(Aindices_, Aindices);

        vec_t_i* r_index_ = new_vector_i(r_index.size());
        copy_vec(r_index_, r_index);

        vec_t* K1_ = new_vector(K1.size());
        copy_vec(K1_, K1);



        t = clock();
        //for (int k = 0; k < bench_iterations; k++) sp_mv_product_vp(Op_, Adata, Aindptr, Aindices, rindex, K1, nrow);
         for (int k = 0; k < bench_iterations; k++) sp_mv_product_vp_(Op_, Adata_, Aindptr_, Aindices_, r_index_, K1_, nrow, nz);
        elapsed_ = 1000 * ((double)(clock() - t)) / (CLOCKS_PER_SEC * bench_iterations_); // time in msec 
        cout << "CSR with omp v1: " << elapsed_ << " msec" << endl;
        //print_vec(Op);

        t_csr_spmv_vp += elapsed_;

        // Version 2.1
        //std::fill(Op.begin(), Op.end(), 0);
        //t = clock();
        //for (int k = 0; k < bench_iterations; k++) sp_mv_product_vp1(Op, Adata, Aindptr, Aindices, rindex, K1);
        //// for(int k=0;k<bench_iterations;k++) sp_mv_product_vp1(Op, Adata, Aindptr, Aindices, rindex, K1);
        //elapsed_ = 1000 * ((double)(clock() - t)) / (CLOCKS_PER_SEC * bench_iterations_); // time in msec 
        //cout << "CSR with omp v2.1 : " << elapsed_ << " msec" << endl;
        // // print_vec(Op);
        //t_csr_spmv_vp1_1 += elapsed_;

        // Version 2.2
        std::fill(Op.begin(), Op.end(), 0);
        t = clock();
        for (int k = 0; k < bench_iterations; k++) sp_mv_product_vp1_(Op_, Adata_, Aindptr_, Aindices_, r_index_, K1_, nrow, nz);
        // for(int k=0;k<bench_iterations;k++) sp_mv_product_vp1(Op, Adata, Aindptr, Aindices, rindex, K1);
        elapsed_ = 1000 * ((double)(clock() - t)) / (CLOCKS_PER_SEC * bench_iterations_); // time in msec 
        cout << "CSR with omp v2.2 : " << elapsed_ << " msec" << endl;
        //print_vec(Op_);
        t_csr_spmv_vp1_2 += elapsed_;

        // version 3.1
        //t = clock();
        //std::fill(Op.begin(), Op.end(), 0);
        //for (int k = 0; k < bench_iterations; k++) sp_mv_product_vp2(Op, Adata, Aindptr, Aindices, rindex, K1);
        ////for (int k = 0; k < bench_iterations; k++) sp_mv_product_vp2_(Op_, Adata_, Aindptr_, Aindices_, r_index_, K1_, nrow, nz);
        //elapsed_ = 1000 * ((double)(clock() - t)) / (CLOCKS_PER_SEC * bench_iterations_); // time in msec 
        //cout << "CSR with omp v3.1 : " << elapsed_ << " msec" << endl;
        // // print_vec(Op);

        //t_csr_spmv_vp2_1 += elapsed_;

        // version 3.2
        t = clock();
        //for (int k = 0; k < bench_iterations; k++) sp_mv_product_vp2(Op, Adata, Aindptr, Aindices, rindex, K1);
        for (int k = 0; k < bench_iterations; k++) sp_mv_product_vp2_(Op_, Adata_, Aindptr_, Aindices_, r_index_, K1_, nrow, nz);
        elapsed_ = 1000 * ((double)(clock() - t)) / (CLOCKS_PER_SEC * bench_iterations_); // time in msec 
        cout << "CSR with omp v3.2 : " << elapsed_ << " msec" << endl;
        //print_vec(Op_);
        t_csr_spmv_vp2_2 += elapsed_;

        // free Memory 

         //cout << "Aindptr_ freeing" << endl;
         free_vector(Aindptr_);
         //cout << "Aindices_ freeing" << endl;
         free_vector(Aindices_);
         //cout << "r_index_ freeing" << endl;
         free_vector(r_index_);
         //cout << "K1_ freeing" << endl;
         free_vector(K1_);
         //cout << "Op_ freeing" << endl;
         free_vector(Op_);
         //cout << "Adata_ freeing" << endl;
         free_vector(Adata_);

        //return 0;
#ifdef _WIN32
        std::ofstream myfile_spmv;
        const std::string dir = "D:\\### PhD Codes ###\\OUTPUTs\\"; //MSI
        std::string  file_name = "spmv_log.txt";
        myfile_spmv.open(dir + file_name, ios::out | ios::app);
#endif
#ifdef linux
        // const std::string dir="//home//ahamsala//scratch//DC//openmp_projects//Output_All//OURS//";
        const std::string dir = "/scratch/ahamsala/DC_CODES/OUTPUTs/COO/";
        std::string file_name = "spmv_log.txt";
        std::string dir_spmv = dir + file_name;
        cout << "linux :: " << dir_spmv.c_str() << endl;
        std::ofstream myfile_spmv;
        myfile_spmv.open(dir_spmv.c_str(), std::ofstream::out | std::ofstream::app);
#endif
        // myfile << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batch\t"<<batch<<"\tdensity\t"<<density<<"\tim2col\t"<<t_im2col<<"\tcsr\t"
        // <<t_csr <<"\tpercent\t"<< 100.0*(t_im2col-t_csr)/t_im2col << "\n";
        if (myfile_spmv.is_open())
        {
            myfile_spmv << Kh << "x" << Kw << " | " << "(" << Ih << "x" << Iw << ")\t"
                << "density\t" << density << '\n'
                << "\t" << t_csr_spmv_product
                << "\t" << t_csr_spmv_seq 
                << "\t" << t_csr_spmv_vp
                //<< "\t" << t_csr_spmv_vp1_1
                << "\t" << t_csr_spmv_vp1_2
                //<< "\t" << t_csr_spmv_vp2_1
                << "\t" << t_csr_spmv_vp2_2
                << "\n";
            cout << '\n' << Kh << "x" << Kw << " | " << "(" << Ih << "x" << Iw << ") batch\t" << batch
                << "\ndensity\t" << density 
                << "\nCSR without omp:\t" << t_csr_spmv_product
                << "\nCSR with element by element without omp:\t" << t_csr_spmv_seq
                << "\nCSR with omp v1  :\t" << t_csr_spmv_vp
                //<< "\nCSR with omp v2  :\t" << t_csr_spmv_vp1_1
                << "\nCSR with omp v2.2  :\t" << t_csr_spmv_vp1_2
                //<< "\nCSR with omp v3.1:\t" << t_csr_spmv_vp2_1
                << "\nCSR with omp v3.2:\t" << t_csr_spmv_vp2_2
                << "\n";
            myfile_spmv.close();
        }
        else
        {
            cout << "Unable to open file";
            return 0;
        }


#endif 


#if Conv_CSR

        // Perform 50 times raw sparse matrix dense vector multiplication: d_o2 = d_m * d_b
        {

            cout << " *** comp scope *** " << endl;

            // cout << " csrMult v2" <<endl;
            cout << "\n csrMult \n" << endl;
            // gettimeofday(&tv1, &tz);
            t_ = clock();
            t = omp_get_wtime();
            // for(int k=0;k<bench_iterations;k++) csrMult(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
            for (int k = 0; k < bench_iterations; k++) csrMult(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
            elapsed_ = 1000 * ((double)(clock() - t_)) / CLOCKS_PER_SEC; // time in msec 
            // gettimeofday(&tv2, &tz);
            cout << "normal csr  --> elapsed time = " << elapsed_ << " msec" << endl;
            d_o2 = d_o2 / bench_iterations;
            t_csr += elapsed_ / (Ih * Iw * 1.0); // normalized timing`
#if IS_PRINT
 // Print out the o1 from im2col:
            std::cout << "\n===CSCC Output with Size: " << d_o2.rows() << ", " << d_o2.cols() << " \n" << d_o2 << std::endl;
            cout << "-----\n" << endl;
#endif     
            // elapsed_ = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
            // double elapsed = 1000*((double)(omp_get_wtime()-t))/CLOCKS_PER_SEC; // time in msec 
        }
        // -------------------------------- Need to be fixed  for OpenMP --------------------------

        {
            cout << "\n csrMult_vp \n" << endl;
            MatrixXf d_o2 = MatrixXf::Zero(Oh, Ow);
            t_ = clock();

            //srMult_vp(MatrixXf& O, vector<double>& K, vector<double>& Adata, vector<int>& Aindices, vector<int>& Aindptr, int Kh, int Kw, int Oh, int
            for (int k = 0; k < bench_iterations; k++) csrMult_vp(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
            elapsed_ = 1000 * ((double)(clock() - t_)) / CLOCKS_PER_SEC; // time in msec
            cout << "Parallel csr v0 --> elapsed time = " << elapsed_ << " msec" << endl;;
            d_o2 = d_o2 / bench_iterations;
            t_csr_vp += elapsed_ / (Ih * Iw * 1.0);
#if IS_PRINT
            // Print out the o1 from im2col:
            std::cout << "\n===CSCC Output with Size: " << d_o2.rows() << ", " << d_o2.cols() << " \n" << d_o2 << std::endl;
            cout << "-----\n" << endl;
#endif
        }
        {
            cout << "\n csrMult_vp1 \n" << endl;
            MatrixXf d_o2 = MatrixXf::Zero(Oh, Ow);
            t_ = clock();
            t = omp_get_wtime();
            for (int k = 0; k < bench_iterations; k++) csrMult_vp1(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
            elapsed = omp_get_wtime() - t;
            elapsed_ = 1000 * ((double)(clock() - t_)) / CLOCKS_PER_SEC; // time in msec
            cout << "Parallel csr v1 --> elapsed time = " << elapsed_ << " msec" << endl;
            d_o2 = d_o2 / bench_iterations;
            t_csr_vp1 += elapsed_ / (Ih * Iw * 1.0);
#if IS_PRINT
            // Print out the o1 from im2col:
            std::cout << "\n===CSCC Output with Size: " << d_o2.rows() << ", " << d_o2.cols() << " \n" << d_o2 << std::endl;
            cout << "-----\n" << endl;
#endif

            // for(int k=0;k<bench_iterations;k++) csrMult_v1(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
            // bench_iterations = 1; // if you want to see the correct result of csr_mult, comment this line
            // for(int k=0;k<bench_iterations;k++) csrMult_v2(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);

        }

        // cout << " do2 :" << d_o2 << "\n do1 :" << d_o1 << endl;
        // MatrixXf sub = d_o2 - d_o2;
        // cout << "Subtraction : " <<  d_o2 - d_o1 <<endl;
       // elapsed time per feature element in the entire bench iterations
        std::cout << "\nbatch = " << In << " --> density = " << density << " // im2col --> " << t_im2col
            << "\ncsr --> " << t_csr << " // csr_vp --> " << t_csr_vp << " // csr_vp1 --> " << t_csr_vp1
            << "\npercent_csr --> " << 100.0 * (t_im2col - t_csr) / t_im2col
            << " // percent_vp --> " << 100.0 * (t_im2col - t_csr_vp) / t_im2col
            << " // percent_vp1 --> " << 100.0 * (t_im2col - t_csr_vp1) / t_im2col
            << std::endl;


        ofstream myfile_csr_conv;

#ifdef _WIN32
        // cout << "_WIN32" << endl;
        myfile_csr_conv.open("D:\\### PhD Codes ###\\OUTPUTs\\csr_conv.txt", ios::out | ios::app);
#endif
#ifdef linux
        // cout << "linux" << endl;
        myfile_csr_conv.open("/home/ahamsala/scratch/DC/openmp_projects/Output_All/OURS/csr_conv.txt", ios::out | ios::app);
#endif
        myfile_csr_conv << Kh << "x" << Kw << " | " << Ih << "x" << Iw << ") batch\t" << batch << "\tdensity\t" << density << "\tim2col\t" << t_im2col << "\tcsr\t"
            << t_csr << " // csr_vp --> " << t_csr_vp << " // csr_vp1 --> " << t_csr_vp1
            << "\npercent_csr --> " << 100.0 * (t_im2col - t_csr) / t_im2col
            << " // percent_vp --> " << 100.0 * (t_im2col - t_csr_vp) / t_im2col
            << " // percent_vp1 --> " << 100.0 * (t_im2col - t_csr_vp1) / t_im2col << "\n";
        myfile_csr_conv.close();

#endif

    } // density loop
    return 0;
}
