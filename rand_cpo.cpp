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


/*SSSSSSSSSSSSSSSSSSSSSSSSSSS*/
void conv_CPO(vector<vector<float> > & O, vector<int> const &K, vector<vector<int> > const &IN,  vector<vector<int> > const &DA, vector<vector<int> > const &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;
    
    int n      = ceil(Kw/Sw);
    int number = floor((Iw - Kw)/Sw) + 1;
    // number = 0; n = 1;
    
    // For each ptr type
    // int type_ptr = 0;
    for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    {
    // For each submat
    for (int submat = 0; submat < number; ++submat)
    {
    
      int shereet2 = (submat == 0)? 0:type_ptr;
      shereet2     = (submat == 1)? 1:shereet2; 
      for(int i = 0; i <= shereet2; ++i)
      {
  
       // From ptr r t r+1
        int shereet = (type_ptr > 0)? 1:0;
  //     int shereet = (type_ptr > 0 and submat > 0)? 1:0;
       // int shereet = 1;

       for(int x = ptr[type_ptr][submat-i*shereet]; x < ptr[type_ptr][submat-i*shereet+1]; ++x)
       {
         
         if(submat-i*shereet < 0)
         {
             cout << "Bug " <<  (submat-i*shereet) << endl;
             int h = 1 + 1;
             exit(0);
         }
         
        // Loop on Kh for the output
       for(int l = 0; l < Kh; ++l)
       {
                    int input_index = IN[type_ptr][x] - i;
        int y_out = (input_index)/Kw - l;
        int x_out = submat;
        
        if(y_out < 0 || y_out >= Oh){
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


void conv_v3(vector<vector<float> > & O, vector<int> const &K, vector<vector<int> > const &IN,  vector<vector<int> > const &DA, vector<vector<int> > const &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
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
      cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " << ptr[type_ptr][submat] <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;
      

      // How many time to iterate?
      int shereet2 = min(submat, type_ptr);
      // int shereet2;
      // if(type_ptr == 0)
      // {
      //   shereet2 = 0;
      // }
      // else
      // {
      //   if(submat == 0)
      //    {
      //     shereet2 = 0;
      //    } 
      //    else
      //    {
      //     if(submat == 1)
      //     {
      //       shereet2 =  1;
      //     }
      //     else
      //     {
      //       shereet2 = type_ptr;
      //     }
           
      //    }
      // }
      
      // int shereet2 = (submat == 0)? 0:type_ptr;
      // shereet2     = (submat == 1)? 1:shereet2; 
      for(int i = 0; i <= shereet2; ++i)
      {
  
       // From ptr r t r+1
        int shereet = (type_ptr > 0)? 1:0;
  //     int shereet = (type_ptr > 0 and submat > 0)? 1:0;
       // int shereet = 1;

       for(int x = ptr[type_ptr][submat-i*shereet]; x < ptr[type_ptr][submat-i*shereet+1]; ++x)
      {

        if(x == ptr[type_ptr][submat-i*shereet])
           cout << "*********ptr_type: " << type_ptr <<  " shereet: " << shereet << " i: "  << i << " submat: " << submat << ", from: " << x << ", to: " << ptr[type_ptr][submat-i+1] << endl;
        else
          cout << "ptr_type: " << type_ptr <<  " shereet: " << shereet << " i: "  << i << " submat: " << submat << ", from: " << x << ", to: " << ptr[type_ptr][submat-i+1] << endl;
        // cout << "x: " << x << ", type_ptr: " << type_ptr << " , submat: " << submat << ", i: " << i << ", shereet: " << shereet << ", v: " << submat-i*shereet << endl ;
         
         if(submat-i*shereet < 0)
         {
             cout << "Bug " <<  (submat-i*shereet) << endl;
             int h = 1 + 1;
             exit(0);
         }
         
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
        
         if(y_out == 0 && x_out == 1)
         cout << "R) " << y_out << ", C) " << x_out << ", Data: " << DA[type_ptr][x] << ", Index: " << input_index  << ", ac_Index: " << IN[type_ptr][x] << endl;
      
         // cout << "R) " << y_out << ", C) " << x_out << ", Data: " << DA[type_ptr][x] << ", Index: " << input_index  << ", ac_Index: " << IN[type_ptr][x] << endl;
    
        // O(y_out, x_out) += DA[type_ptr][x] * 1.0;
        O[y_out][x_out] += DA[type_ptr][x] * K[input_index%Kw + l*Kw];
      
       } // for each l in Kh

      } // for each raga3 el shereet
     } // for each x from range ptr and ptr + 1
  } // for each submat
  
    }

    cout << "Output: " << endl;
    print2DVectorF(O);
    cout << "-----\n" << endl;
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

     cout << "\nPtr: ";
     print2DVector(ptr);

     cout << "\n\nIN: ";
     print2DVector(IN);

     cout << "\n\nData:";
     print2DVector(DA);
     cout << "\n" << endl;

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

  int bench_iterations = 10;
  // int bench_iterations = 2;
  
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





  // for(int i = 0; i < density*Ih*Iw; ++i)
  // {
  //   int r        = rand()%Ih;
  //   int c        = rand()%Iw;
  //   org_fm(r, c) = 1;
  // }
      
  std::cout << "\n===Original Feature Map (" << Ih << "x" << Iw <<  "):  \n" << org_fm << std::endl;
  cout << "-----\n" << endl;

  // Create the filter K and its vectorized version:
  MatrixXf filter             = MatrixXf::Ones(Kh, Kw);
  VectorXf filter_vectorized  = VectorXf::Ones(Kh*Kw);

  std::cout << "\n===Filter: " <<  " \n" << filter_vectorized  << std::endl;
  cout << "-----\n" << endl;

  // Prepare the output for im2col, sparseMat
  MatrixXf d_o1 = MatrixXf::Zero(Oh, Ow);
  MatrixXf d_o2 = MatrixXf::Zero(Oh, Ow);
  
  // Create the Kernel
  vector<int> Kernel(Kh*Kw, 1);

  

  // Prepare the Adata, Aindices, AindPtr for CPO multiplication
   int n = ceil(Kw / Sw);
   if (Kw % Sw == 0)
   {   
        n = Kw / Sw; 
   } 
  
    vector<vector<int> > IN(n); // n is the rows
    vector<vector<int> > DA(n); // n is the rows
    vector<vector<int> > ptr( n , vector<int> (Ow + 1, 0));
    
    CPO(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr);

  // Perform 50 times raw sparse matrix dense vector multiplication: d_CPO = d_m * d_b
   {  
      clock_t t;
      
       for(int k=0;k<bench_iterations;k++){
           // Prepare the output for CPO
           vector<vector<float> > O( Oh , vector<float> (Ow, 0));
           
           t = clock();
           // conv_CPO(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           conv_v3(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
           t_cpo+=elapsed/(Ih*Iw*1.0); // normalized timing

           if(k == bench_iterations-1)
           {
            print2DVectorF(O);
            cout << "-----\n" << endl;
           }
       }
       
      
   }

  // elapsed time per feature element in the entire bench iterations
  // std::cout<<"batch\t"<<In<<"\tdensity\t"<<density <<"\tdensity_cal\t"<<density_cal <<"\tim2col\t"<< t_im2col <<"\tcsr\t"<< t_csr <<"\tcpo\t"<< t_cpo <<std::endl;
    std::cout << "CPO:\t" << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batch\t"<<1
        <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
        <<"\tim2col\t"<<t_im2col<<"\tcsr\t" <<t_csr <<"\tcpo\t"<< t_cpo
         <<"\tpercent1\t"<< 100.0*(t_im2col-t_csr)/t_im2col  <<"\tpercent2\t"<< 100.0*(t_im2col-t_cpo)/t_im2col << "\n";

  
  } // density loop
 
  return 0;
}
