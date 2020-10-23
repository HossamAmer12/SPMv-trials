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
#define case_1          0
#define case_1_time     0
#define encode_all      0
using namespace Eigen;
using namespace std;
using namespace boost::timer;

void printVector(std::vector<int>& x);
// Org Piece:
void CPO(MatrixXf& lowered_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> > &IN,
         vector<vector<int> > &DA, vector<vector<int> > &ptr);

// CPO1 with Ahmad's modifications
int CPO1(MatrixXf& lowered_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> > &IN,
         vector<vector<int> > &DA, vector<vector<int> > &ptr);

// CPO2 with 1-D modifications:
int CPO2(MatrixXf& lowered_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> > &IN,
         vector<vector<int> > &DA, vector<int> &ptr);

// CPO3 with 1-d modifications and removing repititions:
int CPO3(MatrixXf& lowered_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> > &IN,
         vector<vector<int> > &DA, vector<int> &ptr);

int CPO4(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> >& IN,
    vector<vector<int> >& DA, vector<int>& ptr, vector<int>& x, vector<int>& m);

// with 1-d modifications and removing repititions:
int CPO3_(MatrixXf& lowered_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> >& IN,
    vector<vector<int> >& DA, vector<int>& ptr);

int CPO3__(MatrixXf& lowered_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> >& IN,
    vector<vector<int> >& DA, vector<int>& ptr, vector<vector<int> >& ptr2d);

// int CPO3___(MatrixXf& lowered_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> >& IN,
//     vector<vector<int> >& DA, vector<int>& ptr, vector<vector<int> >& ptr2d);
int CPO3___(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> >& IN,
    vector<vector<int> >& DA, vector<int>& ptr, vector<int>& x, vector<int>& m);

void CSR(std::vector<int> &DA, std::vector<int> &IN, std::vector<int> &ptr, MatrixXf& org_mat, int Kh, int Kw, int Oh, 
    int Ow, int Sh, int Sw, int Ih, int Iw);

void CPO5_OP_(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw,
            vector<int>& ptr, vector<int>& IN_1d, vector<int>& DA_1d, vector<int>& x, vector<int>& m);


void CPO5_OP(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw,
            vector<int>& ptr, vector<int>& IN_1d, vector<int>& DA_1d, vector<int>& m);
// https://scicomp.stackexchange.com/questions/27977/how-can-i-speed-up-this-code-for-sparse-matrix-vector-multiplication
bool isErrorCSRInd(std::vector<int>& x1, std::vector<int>& x2, std::vector<double>& y1, std::vector<double>& y2)
{
    bool error = false;

    assert(x1.size() == x2.size());
    assert(y1.size() == y2.size());

    // Indices
    for(int i = 0; i < x1.size(); ++i)
    {
        bool found = false;
        for(int j = 0; j < x2.size(); ++j)
        {
            if(x1[i] == x2[j])
            {
              found = true;
              break;
            }

        }

        if(!found)
        {
          error = true;
          return error;
        }
    }
    

    // Data:
    for(int i = 0; i < y1.size(); ++i)
    {
        bool found = false;
        for(int j = 0; j < y2.size(); ++j)
        {
            if(y1[i] == y2[j])
            {
              found = true;
              break;
            }

        }

        if(!found)
        {
          error = true;
          return error;
        }
    }

  return error;
}


bool isErrorCSRPtr(std::vector<int>& x1, std::vector<int>& x2)
{
    // printVector(x1);
    // printVector(x2);
    assert(x1.size() == x2.size());

    // Indices
    for(int i = 0; i < x1.size(); ++i)
    {
        if(x1[i] != x2[i])
        {
          return true;
        }
    }

    return false;
  
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

void printVector(std::vector<int>& x)
{
    cout << "\nPrint 1D Vector" << endl;
    for(int i = 0; i < x.size(); ++i)
    {
        cout << x[i] << ", ";
    }
    cout << "\n";
}

void printVectorD(std::vector<double>& x)
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


  // Convert the Ptr:
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


  // Convert the data and indices to 1-D:
  for(int i = 0; i < DA.size(); ++i)
  {
    const vector<int> & v1 = DA[i];
    DA_1d.insert( DA_1d.end() , v1.begin() , v1.end());

    const vector<int> & v2 = IN[i];
    IN_1d.insert( IN_1d.end() , v2.begin() , v2.end());
  }

}


void transform2dTo1dV7(vector<vector<int> >  &IN,  vector<vector<int> > &DA, vector<vector<int> >  &ptr, vector<int>  &IN_1d,  vector<int> &DA_1d, vector<int>  &ptr_1d)
{ 


  // Convert the Ptr:
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
          // ptr_1d[c++] = ptr[i][j];
        ptr_1d.push_back(ptr[i][j]);
      }
    }
    else{
      ptr_1d.push_back(ptr[i][j]);
      // ptr_1d[c++] = ptr[i][j];
    } 
  }
 } 

  // Convert the data and indices to 1-D:
  for(int i = 0; i < DA.size(); ++i)
  {
    const vector<int> & v1 = DA[i];
    DA_1d.insert( DA_1d.end() , v1.begin() , v1.end());

    const vector<int> & v2 = IN[i];
    IN_1d.insert( IN_1d.end() , v2.begin() , v2.end());
  }

}


void transform2dTo1d(vector<vector<int> >  &IN,  vector<vector<int> > &DA, vector<int>  &IN_1d,  vector<int> &DA_1d)
{ 
  // Convert the data and indices to 1-D:
    // vector<int>().swap(IN_1d);
    // vector<int>().swap(DA_1d);

    for(int i = 0; i < DA.size(); ++i)
    {
        const vector<int> & v1 = DA[i];
        DA_1d.insert( DA_1d.end() , v1.begin() , v1.end() );

        const vector<int> & v2 = IN[i];
        IN_1d.insert( IN_1d.end() , v2.begin() , v2.end() );

    }

    // DA_1d.erase();
    // IN_1d.erase();


  // Convert the data and indices to 1-D:
  // for(int i = 0; i < DA.size(); ++i)
  // {
  //   const vector<int> v1 = DA[i];
  //   //std::fill( DA_1d.begin() , DA_1d.end(), v1 );
  //   std::move(v1.begin(), v1.end(), std::back_inserter(DA_1d));

  //   const vector<int> v2 = IN[i];
  //   //std::fill( IN_1d.end() , IN_1d.end(), v2);
  //   std::move(v2.begin(), v2.end(), std::back_inserter(IN_1d));

  // }

}

void transform2dTo1d_old(vector<vector<int> >  &IN,  vector<vector<int> > &DA, vector<vector<int> >  &ptr, vector<int>  &IN_1d,  vector<int> &DA_1d, vector<int>  &ptr_1d)
{ 

  // Convert the ptr to 1-D:
  for (int i = 0; i < ptr.size(); ++i)
  {
    const vector<int> & v = ptr[i];
    ptr_1d.insert( ptr_1d.end() , v.begin() , v.end() );
  }

  // Convert the data and indices to 1-D:
  for(int i = 0; i < DA.size(); ++i)
  {
    const vector<int> & v1 = DA[i];
    DA_1d.insert( DA_1d.end() , v1.begin() , v1.end() );

    const vector<int> & v2 = IN[i];
    IN_1d.insert( IN_1d.end() , v2.begin() , v2.end() );

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


void conv_CPO_v9_trim(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<double> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    const int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    const int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;
    
    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int *x_ptr           = &ptr[0];
    int  x               = *x_ptr;
    int *Aindex_help     = &IN[x];
    double *Adata_help      = &DA[x];

    // For each ptr type
    // int type_ptr = 0;
    for (int type_ptr = 0; type_ptr < n-1; ++type_ptr)
    {


      // Submat 0:
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

       // x loop
      // l loop
      // cout << "V7) Sumbat: " << 0 << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    

      // cout << "V9) Sumbat: " << (0) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        double used_data   = *Adata_help;  Adata_help++;

        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int y_out        = (used_index/Kw) - l;

            // 2, 1, 0
            // cout << "Y_out " << y_out << endl;
            if(y_out >= 0 && y_out < Oh) 
            {
               // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
              O[y_out][0] += used_data * K[used_index%Kw + l*Kw];


              // for loop for i and sumbat 0
            int kernel_common_index  = used_index%Kw + l*Kw;

            for(int i = 1; i <= type_ptr; ++i)
            {
               int input_index = used_index - i;
               O[y_out][i] += used_data * K[input_index%Kw + l*Kw];  

              // O[y_out][i] += used_data * K[kernel_common_index - i]; 

            } // end i for loop

          } // end if(y_out >= 0 && y_out < Oh) 

       } // end l loop
      } // end x


      // Submat number - 1:
      x              = *x_ptr; 
      end_x_loop = *(x_ptr+1); 
      ++x_ptr;

      // cout << "V9) Sumbat: " << (number-1) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    

      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        double used_data   = *Adata_help;  Adata_help++;

        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int y_out        = (used_index/Kw) - l;

            // 2, 1, 0
            // cout << "Y_out " << y_out << endl;
            if(y_out >= 0 && y_out < Oh) 
            {
               // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
              O[y_out][number - 1 - type_ptr] += used_data * K[used_index%Kw + l*Kw];


              // for loop for i and sumbat 0
            int kernel_common_index  = used_index%Kw + l*Kw;

            for(int i = 1; i <= type_ptr; ++i)
            {
              // O[y_out][number - 1 - type_ptr + i] += used_data * K[kernel_common_index - i]; 

               int input_index = used_index - i;
               O[y_out][number - 1 - type_ptr + i] += used_data *  K[input_index%Kw + l*Kw];  
        
            } // end i for loop

          } // end if(y_out >= 0 && y_out < Oh) 

       } // end l loop
      } // end x


      ++x_ptr;
    } // end type ptr first loop


    // Last Type ptr
    // for (int type_ptr = 10000; type_ptr < n; ++type_ptr)
    int type_ptr = n-1;
    {

     
    // For each submat
    for (int submat = 0; submat < number; ++submat)
    {
     
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

      // cout << "V9) Sumbat: " << submat << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;   
      // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;

   
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        double used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int input_index  = used_index;
            int y_out        = (input_index/Kw) - l;
            int x_out        = submat;
            int kernel_common_index  = input_index%Kw + l*Kw;

            if(y_out >= 0 && y_out < Oh) {
               O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
               

               for(int i = 1; i <= type_ptr; ++i)
               {


              
                 // Option 1:                 
                 // O[y_out][x_out + i] += used_data * K[kernel_common_index - i]; 

                 // Option 2:
                 int input_index     = used_index - i;
                 O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 

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

// Removing repeats + previous optimizations
void conv_CPO_v7(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<double> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    // printVector(ptr);
    // exit(0);

    int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr

    // number = 0; n = 1;
    
    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int *x_ptr           = &ptr[0];
    int  x               = *x_ptr;
    int *Aindex_help     = &IN[x];
    double *Adata_help   = &DA[x];

    // For each ptr type
    // for (int type_ptr = 0; type_ptr < n; ++type_ptr)
 
   // For each submat
    for (int submat = 0; submat < number; ++submat)
    {

      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

      // cout << "Sumbat: " << submat << ", type_ptr: " << 0 << " start " << x  << " end: " << end_x_loop  << endl;

   
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        int used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int input_index  = used_index;
            int y_out        = (input_index/Kw) - l;
            int x_out        = submat;

            if(y_out >= 0 && y_out < Oh) {
               O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
          } // end if(y_out >= 0 && y_out < Oh) 

       } // end l loop

      } // end x
    } // end sumbat

    

    // cout << "Output: " << endl;
    // print2DVectorF(O);
    // cout << "-----\n" << endl;
}


void conv_CPO_v8_trim(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<double> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    const int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    const int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;
    
    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int *x_ptr           = &ptr[0];
    int  x               = *x_ptr;
    int *Aindex_help     = &IN[x];
    double *Adata_help      = &DA[x];

    // For each ptr type
    // for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    int type_ptr = 0;
    {

       // Submat 0
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

       // x loop
      // l loop
            // cout << "V7) Sumbat: " << 0 << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        double used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int y_out        = (used_index/Kw) - l;

            // 2, 1, 0
            // cout << "Y_out " << y_out << endl;
            if(y_out >= 0 && y_out < Oh) {
               // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
              O[y_out][0] += used_data * K[used_index%Kw + l*Kw];
          } // end if(y_out >= 0 && y_out < Oh) 

       } // end l loop

      } // end x

      // Submat number - 1
      x              = *x_ptr; 
      end_x_loop = *(x_ptr+1); 
      ++x_ptr;


      // cout << "V7) Sumbat: " << (number-1) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
      // x loop
      // l loop
            // cout << "V7) Sumbat: " << (number) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        double used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int y_out        = (used_index/Kw) - l;

            // 2, 1, 0
            // cout << "Y_out " << y_out << endl;
            if(y_out >= 0 && y_out < Oh) {
               // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
              O[y_out][number - 1] += used_data * K[used_index%Kw + l*Kw];
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
     
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      if (end_x_loop<0) {*(x_ptr+1) = x;}
      ++x_ptr;

      // cout << "V7) Sumbat: " << submat << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;   
      // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;

   
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        double used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int input_index  = used_index;
            int y_out        = (input_index/Kw) - l;
            int x_out        = submat;
            int kernel_common_index  = input_index%Kw + l*Kw;

            if(y_out >= 0 && y_out < Oh) {
               O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
               

               for(int i = 1; i <= type_ptr; ++i)
               {


                 int input_index     = used_index - i;
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


void conv_CPO_v7_trim(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<double> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    const int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    const int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;
    
    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int *x_ptr           = &ptr[0];
    int  x               = *x_ptr;
    int *Aindex_help     = &IN[x];
    double *Adata_help      = &DA[x];

    // For each ptr type
    // for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    int type_ptr = 0;
    {

       // Submat 0
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

       // x loop
      // l loop
            // cout << "V7) Sumbat: " << 0 << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        int used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int y_out        = (used_index/Kw) - l;

            // 2, 1, 0
            // cout << "Y_out " << y_out << endl;
            if(y_out >= 0 && y_out < Oh) {
               // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
              O[y_out][0] += used_data * K[used_index%Kw + l*Kw];
          } // end if(y_out >= 0 && y_out < Oh) 

       } // end l loop

      } // end x

      // Submat number - 1
      x              = *x_ptr; 
      end_x_loop = *(x_ptr+1); 
      ++x_ptr;


      // cout << "V7) Sumbat: " << (number-1) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
      // x loop
      // l loop
            // cout << "V7) Sumbat: " << (number) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        int used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int y_out        = (used_index/Kw) - l;

            // 2, 1, 0
            // cout << "Y_out " << y_out << endl;
            if(y_out >= 0 && y_out < Oh) {
               // O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
              O[y_out][number - 1] += used_data * K[used_index%Kw + l*Kw];
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
     
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

      // cout << "V7) Sumbat: " << submat << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;   
      // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;

   
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        int used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int input_index  = used_index;
            int y_out        = (input_index/Kw) - l;
            int x_out        = submat;
            

            if(y_out >= 0 && y_out < Oh) {
               O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];

               for(int i = 1; i <= type_ptr; ++i)
               {
                 int input_index     = used_index - i;
                 O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 

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







// Trying to put everything together in terms of break... loop unrolling for 3x3 filter
void conv_CPO_v6(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<int> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;
    
    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int *x_ptr           = &ptr[0];
    int  x               = *x_ptr;
    int *Aindex_help     = &IN[x];
    int *Adata_help      = &DA[x];

    // For each ptr type
    // for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    int type_ptr = 0;
    {

     
    // For each submat
    for (int submat = 0; submat < number; ++submat)
    {
     
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

      // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;

   
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        int used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int input_index  = used_index;
            int y_out        = (input_index/Kw) - l;
            int x_out        = submat;
            
            if(y_out >= 0 && y_out < Oh) {
               O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
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
     
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

      // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;

   
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        int used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
            int i = 0;
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int input_index  = used_index - i;
            int y_out        = (input_index/Kw) - l;
            int x_out        = i + submat;
            
            if(y_out >= 0 && y_out < Oh) {
               O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];

                int input_index     = used_index - 1;
                O[y_out][x_out + 1] += used_data * K[input_index%Kw + l*Kw];  

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
     
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

      // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;

      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        int used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
            int i = 0;
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int input_index  = used_index - i;
            int y_out        = (input_index/Kw) - l;
            int x_out        = i + submat;
            
            if(y_out >= 0 && y_out < Oh) {
                O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];

                int input_index     = used_index - 1;
                O[y_out][x_out + 1] += used_data * K[input_index%Kw + l*Kw];  

                i = 2;
                input_index     = used_index - 2;
                O[y_out][x_out + 2] += used_data * K[input_index%Kw + l*Kw]; 

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
// Trying to put everything together in terms of break and loop
void conv_CPO_v5(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<int> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;
    
    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int *x_ptr           = &ptr[0];
    int  x               = *x_ptr;
    int *Aindex_help     = &IN[x];
    int *Adata_help      = &DA[x];

    // For each ptr type
    for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    // int type_ptr = 0;
    {

     
    // For each submat
    for (int submat = 0; submat < number; ++submat)
      // for (int submat = 0; submat < 1; ++submat)
    {
     
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

      // cout << "V5) Sumbat: " << 0 << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    

   
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        int used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
            int i = 0;
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int input_index  = used_index - i;
            int y_out        = (input_index/Kw) - l;
            int x_out        = i + submat;
            
            if(y_out >= 0 && y_out < Oh) {

               O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];

               for(int i = 1; i <= type_ptr; ++i)
               {
                 int input_index     = used_index - i;
                 O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 

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



// conv_CPO_v3 for type ptr 2 only
void conv_CPO_v44(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<int> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;
    
    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int *x_ptr           = &ptr[0];
    int  x               = *x_ptr;
    int *Aindex_help     = &IN[x];
    int *Adata_help      = &DA[x];

    // For each ptr type
    // for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    int type_ptr = 2;
    {

     
    // For each submat
    for (int submat = 0; submat < number; ++submat)
    {
     
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

      // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;

        
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        int used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
      
        for(int i = 0; i <= type_ptr; ++i)
        {

          // Loop on Kh for the output
        for(int l = 0; l < Kh; ++l)
        {
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int input_index  = used_index - i;
            int y_out        = (input_index/Kw) - l;
            int x_out        = i + submat;
        
           if(y_out < 0 || y_out >= Oh){
            // cout << "continue YYY============\n" << endl;
            continue;
         }
        
    
        // cout << "R) " << y_out << ", C) " << x_out << ", Data: " << DA[type_ptr][x] << ", Index: " << input_index  << ", ac_Index: " << IN[type_ptr][x] << endl;
       //    // O(y_out, x_out) += DA[type_ptr][x] * 1.0;
          O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
      
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



void Im2col_Encoding(MatrixXf &im2col_mat, MatrixXf& org_fm, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, int Ic)
{
    int start_row_int = 0;
    int start_col_int = 0;
   
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

void reset_Im2col_Encoding(MatrixXf &im2col_mat)
{
    int start_row_int = 0;
    int start_col_int = 0;
   
    for(int i = 0; i < im2col_mat.rows() ; ++i)
    {
      for(int j = 0; j < im2col_mat.cols(); ++j)
      {
        im2col_mat(i, j) = 0;
      }
    }
}

void reset_ptr(std::vector <int>& x, int num = 0)
{

    for (int i = 0; i < x.size(); ++i)
    {
        x[i] = num;
    }
}




double generate_org_featureMap(MatrixXf& org_fm, int Ih, int Iw, double density)
{
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
    return density_cal;

}

int CPO_Encoding(std::vector<int> &IN_1d, std::vector<int> &DA_1d, std::vector<int> &ptr_1d, MatrixXf& org_fm, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
  int n = ceil(Kw / Sw);
  if (Kw % Sw == 0)
  {
      n = Kw / Sw;
  }
  
  // transform 2d to 1d:
  int count_ptr = n*(1+Ow);

  vector<vector<int> > IN(n); // n is the rows
  vector<vector<int> > DA(n); // n is the rows
  
  // int count_d = CPO1(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr);

  int count_d = CPO2(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr_1d);
  IN_1d.reserve(count_d);
  DA_1d.reserve(count_d);  
  transform2dTo1d(IN, DA, IN_1d, DA_1d);
  return count_d;

}


int CPO3_(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> >& IN,
    vector<vector<int> >& DA, vector<int>& ptr)
{
    // cout << "########################### CPO3_ ###########################" << endl;

    int l = 0;
    int n = Kw;
    int non_zero_count = 0;
    int npo_shift = (Ow+1) - 3; 


    std::vector<int> x(n, 0);
    std::vector<int> m(n, 0);

    // NPO
    {
        int j = 0;
        ptr[l * (1 + Ow) + m[l]] = x[l];
        m[l]++;
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                int ind_val = j + (i * Kw); 
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
                non_zero_count++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)       
        ptr[l * (1 + Ow) + m[l]] = x[l];
        m[l]++;
        l++;
    }
    // first piece
    for (int j = 1; j < Kw; ++j)
    {
        ptr[l * (1 + Ow) + m[l] - npo_shift] = x[l];
        m[l]++;
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                int ind_val = j + (i * Kw); 
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
                non_zero_count++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)       
        ptr[l * (1 + Ow) + m[l] - npo_shift] = x[l];
        m[l]++;
        l++;
    } // end for (int j = 0; j < Kw; ++j)

    l--;
    // cout << "before second : ";
    // printVector(m);
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
                non_zero_count++;
            } // end if (org_mat(i, j) != 0)
        }// end for(i = 0; i < Ih; ++i)
        ptr[l * (1 + Ow) + m[l] - npo_shift] = x[l];
        m[l]++;
        // cout << "CPO3_wo_rep x_helper -- > " << x[l] << endl;
        if (n > 1)
        {
            for (int c = 1; c < n - 1; ++c)
                // for (int c = 1; c < 4; ++c)
            {
                ptr[c * (1 + Ow) + m[c] - npo_shift] = x[c];
                m[c]++;
                non_zero_count++;
            } // end for(int c = 0; c < n-1; ++c)
        } // end if(n > 1) 
    } // end for(int j = Kw; j < Iw - Kw; ++j)

     // cout << "before the 3rd piece : " << endl;
     // printVector(m);
    // Third piece   
    for (int j = Iw - Kw; j < Iw - 1; ++j)
    {
        // cout << l <<" Img col : " << j << endl;
        for (int i = 0; i < Ih; i++)
        {
            if (org_mat(i, j) != 0)
            {
                int ind_val = j - (m[l] - 1) + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++; // accum on the last value of the pointer 
                non_zero_count++;

            }// end if(org_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)
        // cout << "Inside for (int c = 0 : " << j << endl;
        for (int c = 0; c < l + 1; ++c)
        {
            // this updates the pointer with new value from the above .. and it is the source of repetition 
            // as it is repeated for each col in the pointer with the same value in a linear way .. where if 
            // it is removed .. no need to use this stage 
            // thinking about stride of 1 and we have only 3 pointers 
            // cout << j <<" -- 3rd a -- ptr : " << l * (1 + Ow) + m[l] << endl;
            ptr[l * (1 + Ow) + m[l] - npo_shift] = x[l];
            // cout << "ptr index : " << l * (1 + Ow) + m[l] - 3 << endl;
            // if are gointg to remove this stage 
            // m[l] +=l;
            m[l]++; // update the col index of the l poiner type 
            // cout << m[l] << endl;
        } // end for(int c = 0; c < l+1; ++c)
        if (l >= 1) // do 
        {
            for (int c = 1; c < l ; ++c)
            {
                // cout << j <<" -- 3rd b (l>=1) --  ptr : " << l * (1 + Ow) + m[l] << endl;
                ptr[c * (1 + Ow) + m[c] - npo_shift] = x[c];
                m[c]++;
            } // end for(int c = 0; c < l-1; ++c)
        }// end if l > 1
        // cout << "Inside if (l >= 1): " << j << endl;
        // printVector(m);
        l--;
    } // end for (j = Iw - Kw; Iw; ++j)
    // NPO
    {
        int j = Iw - 1;
        for (int i = 0; i < Ih; i++)
        {
            if (org_mat(i, j) != 0)
            {
                int ind_val = Kw - 1 + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
                non_zero_count++;

            }// end if(org_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)

        ptr[2] = x[l];
    }

#if 0
   // std::cout << "\nPtr: ";
  // print2DVector(ptr);

   // std::cout << "\n\nIN: ";
  // print2DVector(IN);

   //  std::cout << "\n\nData:";
    //  print2DVector(DA);
   // std::cout << "\n" << endl;
#endif 
    // cout << "########################### END CPO3_ ###########################" << endl;
    return non_zero_count;
}


int CPO3__(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> >& IN,
    vector<vector<int> >& DA, vector<int>& ptr, vector<vector<int> >& ptr2d)
{
    // cout << "Start" << endl;

    int l = 0;
    int n = Kw;
    int non_zero_count = 0;
    int npo_shift = (Ow+1) - 3; 

    int ind_val;
    std::vector<int> x(n, 0);
    std::vector<int> m(n, 0);
    int ptr_help;
    // NPO
    // m[l] is not important 
    {
        int j = 0;
        // ptr2d[l][m[l]] = x[l];
        ptr[l * (1 + Ow) + m[l]] = x[l];
        m[l]++;
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                ind_val = j + (i * Kw); 
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
                non_zero_count++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)       
        // ptr2d[l][m[l]] = x[l];
        ptr[l * (1 + Ow) + m[l]] = x[l];
        m[l]++;
        l++;
    }
    // first piece
    for (int j = 1; j < Kw; ++j)
    {
        // ptr2d[l][m[l]] = x[l];
        ptr[l * (1 + Ow) + m[l] - npo_shift] = x[l];
        m[l]++;
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                ind_val = j + (i * Kw); 
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
                non_zero_count++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)       
        // ptr2d[l][m[l]] = x[l];
        ptr[l * (1 + Ow) + m[l] - npo_shift] = x[l];
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
                ind_val = j - (m[l] - 1) + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
                non_zero_count++;
            } // end if (org_mat(i, j) != 0)
        }// end for(i = 0; i < Ih; ++i)
        // ptr2d[l][m[l]] = x[l];
        ptr[l * (1 + Ow) + m[l] - npo_shift] = x[l];
        m[l]++;
 
    } // end for(int j = Kw; j < Iw - Kw; ++j)

    // increment the position of col in the pointer of PO2
    // repeating elements for all pointers except NPO//last pointer
    for (int c = 1; c < n - 1; ++c)
    {
        // ptr2d[c][m[c]] = x[c];
        // ptr[c * (1 + Ow) + m[c] - npo_shift] = x[c];
        m[c] += Iw - 2*Kw + 1;
    } // end for(int c = 0; c < n-1; ++c)
    
    // cout << "\nafter stage 2 " << endl;
    // printVector(ptr);

    // Third piece   
    for (int j = Iw - Kw; j < Iw - 1; ++j)
    {
        // cout << l <<" Img col : " << j << endl;
        for (int i = 0; i < Ih; i++)
        {
            if (org_mat(i, j) != 0)
            {
                ind_val = j - (m[l] - 1) + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++; // accum on the last value of the pointer 
                non_zero_count++;

            }// end if(org_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)

        ptr[l * (1 + Ow) + m[l] - npo_shift] = x[l];
        m[l] += l;
        l--;
    } // end for (j = Iw - Kw; Iw; ++j)
    // NPO
    {
        int j = Iw - 1;
        for (int i = 0; i < Ih; i++)
        {
            if (org_mat(i, j) != 0)
            {
                ind_val = Kw - 1 + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
                non_zero_count++;

            }// end if(org_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)
        ptr[2] = x[l];
    }

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

}

int CPO3___(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> >& IN,
    vector<vector<int> >& DA, vector<int>& ptr, vector<int>& x, vector<int>& m)
{
    // cout << "Start" << endl;
    int Ow_1 = (1 + Ow);
    // int l = 1; // Kw-1
    int l = Kw-1; // Kw-1
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
        ptr_Indexhelper = j * Ow_1 - npo_shift + 1 ; 
        ptr[ptr_Indexhelper - 1] =  0;
        ptr[ptr_Indexhelper    ] =  x_helper1;
        x[j]                     =  x_helper1;
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
                    ind_val = j - ( m_helper - 1) + (i * Kw);
                    IN[l].push_back(ind_val);
                    DA[l].push_back(org_mat(i, j));
                    // x[l]++;
                    // (*x_ptr)++;
                    x_helper++;
                    non_zero_count++;
                } // end if (org_mat(i, j) != 0)
            }// end for(i = 0; i < Ih; ++i)

            ptr[ ptr_Indexhelper + m_helper] = x_helper;
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

    // Third piece   
    for (int j = Iw - Kw; j < Iw - 1; ++j)
    {
        x_helper = x[l];
        m_helper = m[l];
        // cout << l <<" Img col : " << j << endl;
        for (int i = 0; i < Ih; i++)
        {
            if (org_mat(i, j) != 0)
            {
                ind_val = j - (m_helper - 1) + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x_helper++; // accum on the last value of the pointer 
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
        for (int c = 0; c < l + 1; ++c)
        {
            // ptr[l][m[l]] = x[l];
            ptr[l_Ow_1 +  m_helper ] = x_helper;
            m_helper++;
        } // end for(int c = 0; c < l+1; ++c)
        if (l >= 1)
        {
            for (int c = 1; c < l; ++c)
            {
                // ptr[c][m[c]] = x[c];
                ptr[c*Ow_1 +  m[c]- npo_shift] = x[c];
                m[c]++;
            } // end for(int c = 0; c < l-1; ++c)
        }// end if l > 1
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

}

int CPO3(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> >& IN,
    vector<vector<int> >& DA, vector<int>& ptr, vector<int>& x, vector<int>& m)
{
    // cout << "Start" << endl;
    int Ow_1 = Ow + 1;
    int l = 0;
    int n = Kw;
    int non_zero_count = 0;
    int npo_shift = Ow_1 - 3; 

    int ind_val;

    int ptr_Indexhelper, x_helper, m_helper;
    // int *m_ptr, *x_ptr;
    // NPO
    // m[l] is not important 
    {
        int j = 0;
        // ptr2d[l][m[l]] = x[l];
        // ptr[l * (1 + Ow) + m[l]] = x[l];

        int *m_ptr = &m[l];
        int *x_ptr = &x[l];

        // m[l]++;
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                ind_val = j + (i * Kw); 
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                // x[l]++;
                (*x_ptr)++;
                non_zero_count++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)       
        // ptr2d[l][m[l]] = x[l];
        ptr[0] = 0;
        ptr[1] = (*x_ptr);
        (*m_ptr) = 2;
        l++;
    }
    // first piece all pointer except NPO 
    for (int j = 1; j < Kw; ++j)
    {
        x_helper = x[l];
        // ptr2d[l][m[l]] = x[l];
        int *x_ptr = &x[l];
        int *m_ptr = &m[l];
        ++(*m_ptr);
        // ptr[l * (1 + Ow) + m[l] - npo_shift] = x[l];
        // m[l]++;
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                ind_val = j + (i * Kw); 
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                // x[l]++;
                ++(*x_ptr);
                non_zero_count++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)
        // #S7es# can be optimized
        ptr_Indexhelper = l * Ow_1 - npo_shift + (*m_ptr);
        ptr[ptr_Indexhelper - 1] = x_helper;
        ptr[ptr_Indexhelper] = (*x_ptr);
        ++(*m_ptr);
        l++;
    } // end for (int j = 0; j < Kw; ++j)

    l--;

    {
        // Second piece
        int *x_ptr = &x[l];
        int *m_ptr = &m[l];
        int *ptr_IndHelper =  &ptr[l * Ow_1 - npo_shift + (*m_ptr) ];
        for (int j = Kw; j < Iw - Kw; ++j)
        {
            for (int i = 0; i < Ih; ++i)
            {
                if (org_mat(i, j) != 0)
                {
                    ind_val = j - ( (*m_ptr) - 1) + (i * Kw);
                    IN[l].push_back(ind_val);
                    DA[l].push_back(org_mat(i, j));
                    (*x_ptr)++;
                    non_zero_count++;
                } // end if (org_mat(i, j) != 0)
            }// end for(i = 0; i < Ih; ++i)
            // ptr2d[l][m[l]] = x[l];
            // ptr[ptr_Indexhelper + (*m_ptr) ] = x[l];
            *ptr_IndHelper = *x_ptr;
            *(++ptr_IndHelper);
            ++(*m_ptr);

        } // end for(int j = Kw; j < Iw - Kw; ++j)
    }


    
    // increment the position of col in the pointer of PO2
    // repeating elements for all pointers except NPO//last pointer
    for (int c = 1; c < n - 1; ++c)
    {
        // ptr2d[c][m[c]] = x[c];
        // ptr[c * (1 + Ow) + m[c] - npo_shift] = x[c];
        m[c] += Iw - 2*Kw + 1;
    } // end for(int c = 0; c < n-1; ++c)
    
    // cout << "\nafter stage 2 " << endl;
    // printVector(ptr);

    // Third piece   
    for (int j = Iw - Kw; j < Iw - 1; ++j)
    {
        int *x_ptr = &x[l];
        int *m_ptr = &m[l];
        // cout << l <<" Img col : " << j << endl;
        for (int i = 0; i < Ih; i++)
        {
            if (org_mat(i, j) != 0)
            {
                ind_val = j - ((*m_ptr) - 1) + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                ++(*x_ptr); // accum on the last value of the pointer 
                non_zero_count++;

            }// end if(org_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)

        ptr[l * Ow_1 + (*m_ptr) - npo_shift] = *x_ptr;
        (*m_ptr) += l;
        l--;
    } // end for (j = Iw - Kw; Iw; ++j)
    // NPO
    {
        int *x_ptr = &x[l];
        int j = Iw - 1;
        for (int i = 0; i < Ih; i++)
        {
            if (org_mat(i, j) != 0)
            {
                ind_val = Kw - 1 + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                (*x_ptr )++;
                non_zero_count++;

            }// end if(org_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)
        ptr[2] = *x_ptr ;
        
    }

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

}



int CPO4(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> >& IN,
    vector<vector<int> >& DA, vector<int>& ptr, vector<int>& x, vector<int>& m)
// (org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr_1d, x, m)
{
     // cout << "########################### CPO4 ###########################" << endl;
    // cout << "Start" << endl;
    int Ow_1 = (1 + Ow);
    int l_Ow_1;
    int l = Kw-1; // Kw-1  Jump to the second part 
    int non_zero_count = 0;
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
                IN[0].push_back(ind_val);
                DA[0].push_back(org_mat(i, 0));
                // x[l]++;
                x_helper++;
                non_zero_count++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)       
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
        // x[0] = x_helper; // no need as we have finished the NPO ... so no need to save it
    }

    // first piece
    int x_helper1;

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
        ptr_Indexhelper = j * Ow_1 - npo_shift + 1 ; 
        ptr[ptr_Indexhelper - 1] =  0;
        ptr[ptr_Indexhelper    ] =  x_helper1;
        x[j]                     =  x_helper1;
    } // end for (int j = 0; j < Kw; ++j)

    // cout << "\n*After 1st Part  m,x : ";
    // printVector(m);
    // printVector(x);
    // printVector(ptr);
    // l--;

    {
        // Second piece
        ptr_Indexhelper = l * Ow_1 - npo_shift;
        // m_helper = m[l];
        m_helper = 2; // always the second element in the pointer 
        x_helper = x[l];

        for (int j = Kw; j < Iw - Kw; ++j)
        {
            for (int i = 0; i < Ih; ++i)
            {
                if (org_mat(i, j) != 0)
                {
                    // ind_val = j - ( (*m_ptr) - 1) + (i * Kw);
                    ind_val = j - ( m_helper - 1) + (i * Kw);
                    IN[l].push_back(ind_val);
                    DA[l].push_back(org_mat(i, j));

                    x_helper++;
                    non_zero_count++;
                } // end if (org_mat(i, j) != 0)
            }// end for(i = 0; i < Ih; ++i)

            // kernel size big ... pointer 
            // accessing is better at small size 
            ptr[ ptr_Indexhelper + m_helper] = x_helper; 
            m_helper++;
            // cout << "CPO4 x_helper -- > " << x_helper << endl;

     
        } // end for(int j = Kw; j < Iw - Kw; ++j)
        x[l] = x_helper;
        m[l] = m_helper;
    }

    // cout << "\n*After 2nd Part m,x: ";
    // printVector(m);
    // printVector(x); 
    // printVector(ptr); 

    // cout << "\nStart 3rd Piece : " << endl;
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
                // cout << "( " << i << " , " << j << " ) --> m: " << m_helper 
                //      << ", idx: " << ind_val << ", l: " << l << endl;

            }// end if(org_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)

        l_Ow_1 = l * Ow_1 - npo_shift;
        ptr[l_Ow_1 +  m_helper ] = x_helper1; 
        for (int c = 1; c < l; ++c)
        {
            m[c]++;
        } // end for(int c = 0; c < l-1; ++c)
        
        x[l] = x_helper1;
        l--;

        // cout << "\nEnd 3rd Part --> " << j;
        // printVector(m);
        // printVector(x);
        // printVector(ptr);  

    } // end for (j = Iw - Kw; Iw; ++j)
     // cout << "########################### END CPO4 ###########################" << endl;
    return non_zero_count;

} //CPO4 


int CPO5(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> >& IN,
    vector<vector<int> >& DA, vector<int>& ptr, vector<int>& IN_1d, vector<int>& DA_1d, vector<int>& x, vector<int>& m)
// (org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr_1d, x, m)
{
    // cout << "Start" << endl;
    int Ow_1 = (1 + Ow);
    int l_Ow_1;
    int l = Kw-1; // Kw-1  Jump to the second part 
    int non_zero_count = 0;
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
                //IN[]
                IN[0].push_back(ind_val);
                DA[0].push_back(org_mat(i, 0));

                IN_1d.push_back(ind_val);
                DA_1d.push_back(org_mat(i, 0));
                // x[l]++;
                x_helper++;
                non_zero_count++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)       
        ptr[0] = 0;
        ptr[1] = x_helper;

        for (int i = 0; i < Ih; i++)
        {
            if (org_mat(i, Iw - 1) != 0)
            {
                ind_val = Kw - 1 + (i * Kw);
                IN[0].push_back(ind_val);
                DA[0].push_back(org_mat(i, 0));

                IN_1d.push_back(ind_val);
                DA_1d.push_back(org_mat(i, Iw - 1));
                x_helper++;
                non_zero_count++;

            }// end if(org_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)
        ptr[2] = x_helper;
        // x[0] = x_helper; // no need as we have finished the NPO ... so no need to save it
    }

   
    // first piece
    int x_helper1, x_helper_;
    // it should start from left to right ... so we need anther incremental for the last part 
    // of the image

    int j_ = Iw - 2;
    l = 1;

    for (int j = 1 ; j < Kw; ++j)
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
                IN[j].push_back(ind_val);
                DA[j].push_back(org_mat(i, j));
                IN_1d.push_back(ind_val);
                DA_1d.push_back(org_mat(i, j));
                // x[l]++;
                // (*x_ptr)++;
                x_helper1++;
                non_zero_count++;
            } // end if  if (org_mat(i, j) != 0)

        }
            // ptr_Indexhelper = l * Ow_1 - npo_shift + m_helper;
        
        // m_helper no need for it as all the pointers will be incremented by 1
        // ptr_Indexhelper = l * (1 + Ow) - npo_shift + m_helper ( 1 );
        ptr_Indexhelper = j * Ow_1 - npo_shift + 1 ; 
        ptr[ptr_Indexhelper - 1] =  0;
        ptr[ptr_Indexhelper    ] =  x_helper1;
        x[j]                     =  x_helper1;
        // cout << "current : " << j << "\t" << l << endl;
#if encode_all
        cout << "\n Start --> 3rd Part --> " << j;
        printVector(m);
        printVector(x);
#endif 
        // Second part
        // x_helper_ = x[l];
        x_helper_ = x_helper1;
        m_helper = m[l];
        for (int i = 0; i < Ih; ++i)
        {

            if (org_mat(i, j_) != 0)
            {
                ind_val = j_ - (m_helper - 1) + (i * Kw);
#if encode_all
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j_));
                cout << "( " << i << " , " << j_ << " ) --> m: " << m_helper 
                     << ", idx: " << ind_val << ", l: " << l << endl;
#endif
                IN_1d.push_back(ind_val);
                DA_1d.push_back(org_mat(i, j_));
                x_helper_++; // accum on the last value of the pointer 
                non_zero_count++;

            }// end if(org_mat(i, j) != 0)

        } // end for(i=0; i < Ih; ++i)       

        l_Ow_1 = l * Ow_1 - npo_shift;
        ptr[l_Ow_1 +  m_helper ] = x_helper_; 
        for (int c = 1; c < l; ++c)
        {
            m[c]--;
        } // end for(int c = 0; c < l-1; ++c)
        x[l] = x_helper_;

        j_--;
        l++;
#if encode_all
        cout << "\n End --> 3rd Part --> " << j;
        printVector(m);
        printVector(x);
#endif
    } // end for (int j = 0; j < Kw; ++j)
    // l--;

    l = Kw-1;

    {
#if encode_all
        cout << " 2nd Piece " << endl;
#endif
        // Second piece
        ptr_Indexhelper = l * Ow_1 - npo_shift;
        // m_helper = m[l];
        m_helper = 2; // always the second element in the pointer 
        x_helper = x[l];

        for (int j = Kw; j < Iw - Kw; ++j)
        {
            for (int i = 0; i < Ih; ++i)
            {
                if (org_mat(i, j) != 0)
                {
                    // ind_val = j - ( (*m_ptr) - 1) + (i * Kw);
                    ind_val = j - ( m_helper - 1) + (i * Kw);
                    IN[l].push_back(ind_val);
                    DA[l].push_back(org_mat(i, j));

                    IN_1d.push_back(ind_val);
                    DA_1d.push_back(org_mat(i, j));
#if encode_all
                    cout << "( " << i << " , " << j << " ) --> m: " << m_helper 
                         << ", idx: " << ind_val << ", l: " << l << endl;
#endif
                    x_helper++;
                    non_zero_count++;
                } // end if (org_mat(i, j) != 0)
            }// end for(i = 0; i < Ih; ++i)

            // kernel size big ... pointer 
            // accessing is better at small size 
            ptr[ ptr_Indexhelper + m_helper] = x_helper; 
            m_helper++;

     
        } // end for(int j = Kw; j < Iw - Kw; ++j)
        // Not important 
        // x[l] = x_helper;
        // m[l] = m_helper;
    }
#if encode_all
    cout << "\n #### CPO5 Index #### \n" << endl;
    printVector(IN_1d);
    print2DVector(IN);
    exit(0);
#endif
    
    return non_zero_count;

} //CPO5

void CPO5_OP(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw,
            vector<int>& ptr, vector<int>& IN_1d, vector<int>& DA_1d, vector<int>& m)
{
    int Ow_1 = (1 + Ow);
    int l_Ow_1;
    int l = 1; // Kw-1  Jump to the second part 
    int npo_shift = Ow_1 - 3;  // as we igorned the repetition i
    int ind_val;
    // Initial value of the index of the pointer by 2 as all the pointer will reach the 
    // same index after the secind piece 
    vector<int> DA_1d_1, IN_1d_1;
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
    }
    // first piece & third piece 
    // it should start from left to right ... so we need anther incremental for the last part 
    // of the image
    int j_ = Iw - 2;
    for (int j = 1 ; j < Kw-1; ++j)
    {
        // First part
        x_helper = 0;
        // m_helper++;
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                ind_val = j + (i * Kw);
                IN_1d.push_back(ind_val);
                DA_1d.push_back(org_mat(i, j));
                x_helper++;
            } // end if  if (org_mat(i, j) != 0)

        }
        
        // m_helper no need for it as all the pointers will be incremented by 1
        // ptr_Indexhelper = l * (1 + Ow) - npo_shift + m_helper ( 1 );
        ptr_Indexhelper = j * Ow_1 - npo_shift + 1 ; 
        ptr[ptr_Indexhelper - 1] =  0;
        ptr[ptr_Indexhelper    ] =  x_helper;
        // Second part
        m_helper = m[l];
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j_) != 0)
            {                 
                ind_val = j_ - (m_helper - 1) + (i * Kw);
                // IN_1d.push_back(ind_val);
                // DA_1d.push_back(org_mat(i, j_));
                IN_1d.push_back(ind_val);
                DA_1d.push_back(org_mat(i, j_));
                x_helper++; // accum on the last value of the pointer 
            }// end if(org_mat(i, j) != 0)

        } // end for(i=0; i < Ih; ++i)       

        l_Ow_1 = l * Ow_1 - npo_shift;
        ptr[l_Ow_1 +  m_helper ] = x_helper; 
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
} //CPO

void CPO5_OP_(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw,
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
    vector<int> DA_1d_1, IN_1d_1;
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
        // ptr_Indexhelper = l * (1 + Ow) - npo_shift + m_helper ( 1 );
        ptr_Indexhelper = j * Ow_1 - npo_shift + 1 ; 
        ptr[ptr_Indexhelper - 1] =  0;
        ptr[ptr_Indexhelper    ] =  x_helper1;
        // x[j]                     =  x_helper1;
        // Second part
        x_helper2 = x_helper1;
        m_helper = m[l];
        for (int i = 0; i < Ih; ++i)
        {

            if (org_mat(i, j_) != 0)
            {
                // cout << "( " << i << " , " << j_ << " ) --> m: " << m_helper 
                //      << ", idx: " << ind_val << ", l: " << l << endl;                     
                ind_val = j_ - (m_helper - 1) + (i * Kw);
                // IN_1d.push_back(ind_val);
                // DA_1d.push_back(org_mat(i, j_));
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
        // x[l] = x_helper2;
        // cout << "*m" << endl;
        // printVector(m);

        j_--;
        l++;
    } // end for (int j = 0; j < Kw; ++j)
    
    // l--;
    // l = Kw-1;
    // cout << "3rd part merged" << endl;
    // cout << "last x_helper1 : " << x_helper1 << endl;
    // printVector(ptr);
    {
        // Second piece
        ptr_Indexhelper = l * Ow_1 - npo_shift;
        ptr[ptr_Indexhelper] = 0;
        // m_helper = m[l];
        m_helper = 1; // always the second element in the pointer 
        // x_helper = x[l];
        // x_helper = x_helper1;
        x_helper = 0;

        for (int j = Kw-1; j <= Iw - Kw; ++j)
        {
            for (int i = 0; i < Ih; ++i)
            {
                if (org_mat(i, j) != 0)
                {
                    // ind_val = j - ( (*m_ptr) - 1) + (i * Kw);
                    // cout << "( " << i << " , " << j << " ) --> m: " << m_helper << ", idx: " << ind_val << ", l: " << l << endl;
                    
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
            // cout << l <<" : CPO5 x_helper -- > " << x_helper << endl;
        } // end for(int j = Kw; j < Iw - Kw; ++j)
        // cout << "CPO5  ptr :\n" << endl;
        // printVector(ptr);
        // ptr[ ptr_Indexhelper + m_helper] = x_helper + x_helper_ - ptr[ ptr_Indexhelper + 1];
        // ptr[ ptr_Indexhelper + m_helper] = x_helper + x_helper2 - x_helper1;
        // x[l] = x_helper;
        // m[l] = m_helper;
    }
    // cout << "\n\nlast ptr" << endl;
    // printVector(ptr);
    // cout << " ############################### END CPO 5 ###############################" << endl;
} //CPO5



int CPO4_(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> >& IN,
    vector<vector<int> >& DA, vector<int>& ptr, vector<int>& x, vector<int>& m)
// (org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr_1d, x, m)
{
    // cout << "Start" << endl;
    int Ow_1 = (1 + Ow);
    int l_Ow_1;
    int l = Kw-1; // Kw-1
    int non_zero_count = 0;
    int npo_shift = Ow_1 - 3; 
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
                IN[0].push_back(ind_val);
                DA[0].push_back(org_mat(i, 0));
                // x[l]++;
                x_helper++;
                non_zero_count++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)       
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
        ptr_Indexhelper = j * Ow_1 - npo_shift + 1 ; 
        ptr[ptr_Indexhelper - 1] =  0;
        ptr[ptr_Indexhelper    ] =  x_helper1;
        x[j]                     =  x_helper1;
    } // end for (int j = 0; j < Kw; ++j)

    // l--;

    {
        // Second piece
        ptr_Indexhelper = l * Ow_1 - npo_shift;
        // m_helper = m[l];
        m_helper = 2;
        x_helper = x[l];

        for (int j = Kw; j < Iw - Kw; ++j)
        {
            for (int i = 0; i < Ih; ++i)
            {
                if (org_mat(i, j) != 0)
                {
                    // ind_val = j - ( (*m_ptr) - 1) + (i * Kw);
                    ind_val = j - ( m_helper - 1) + (i * Kw);
                    IN[l].push_back(ind_val);
                    DA[l].push_back(org_mat(i, j));

                    x_helper++;
                    non_zero_count++;
                } // end if (org_mat(i, j) != 0)
            }// end for(i = 0; i < Ih; ++i)

            // kernel size big ... pointer 
            // accessing is better at small size 
            ptr[ ptr_Indexhelper + m_helper] = x_helper; 
            m_helper++;

     
        } // end for(int j = Kw; j < Iw - Kw; ++j)
        x[l] = x_helper;
        m[l] = m_helper;
    }

    

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

        l_Ow_1 = l * Ow_1 - npo_shift;
        ptr[l_Ow_1 +  m_helper ] = x_helper1; 
        for (int c = 1; c < l; ++c)
        {
            m[c]++;
        } // end for(int c = 0; c < l-1; ++c)
        
        l--;

    } // end for (j = Iw - Kw; Iw; ++j)

    return non_zero_count;

} //CPO4 


void CPO_EncodingV7(std::vector<int>& IN_1d, std::vector<int>& DA_1d, std::vector<int>& ptr_1d, MatrixXf& org_fm, int Kh,
 int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw,  std::vector<vector<int>>& IN,  std::vector<vector<int>>& DA, 
 std::vector<int>& x, std::vector<int>& m, int n )
// int CPO_EncodingV7(std::vector<int>& ptr_1d, MatrixXf& org_fm, int Kh,
//  int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw,  std::vector<vector<int>>& IN,  std::vector<vector<int>>& DA, 
//  std::vector<int>& x, std::vector<int>& m, int n )
{

  // Get the total number of non zeros: we can save it while encoding:
    if (n != 1 )
    {
        // the only thing done here is using of 1d pointer instead of 2d
        
        int count_d  = CPO4(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr_1d, x, m);
        IN_1d.reserve(count_d);
        DA_1d.reserve(count_d);
        // transform2dTo1d(IN, DA, IN_1d, DA_1d, ptr_1d);
        transform2dTo1d(IN, DA, IN_1d, DA_1d);
    }
    else 
    {
        CSR(DA_1d, IN_1d, ptr_1d, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
    }

}

void CPO_EncodingV8(std::vector<int>& IN_1d, std::vector<int>& DA_1d, std::vector<int>& ptr_1d, MatrixXf& org_fm, int Kh,
 int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, std::vector<int>& m, int n )
{
    // Get the total number of non zeros: we can save it while encoding:
    if (n != 1 )
    {
        for (int i = 1 ; i < m.size()-1 ; i++)
        {
            m[i] = Ow - i;
        }
        // the only thing done here is using of 1d pointer instead of 2d
        CPO5_OP(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, ptr_1d, IN_1d, DA_1d, m);
    }
    else 
    {
        CSR(DA_1d, IN_1d, ptr_1d, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
    }
}


void All_CPO(std::vector<int>& IN_1d, std::vector<int>& DA_1d, std::vector<int>& ptr_1d, MatrixXf& org_fm, int Kh, int Kw, 
    int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
    cout << " *********************** All Ecoding *******************************" << endl;
    // int bench_iterations = 100000;
    int bench_iterations = 1;
    int n = ceil(Kw / Sw);
    if (Kw % Sw == 0)
    {
        n = Kw / Sw;
    }
    // bench_iterations = 100000;

    float t_cpo_creation_V7 = 0;
    float t_cscc_creation = 0;
    cout << " ####### CPO ####### " << endl;
    for (int k = 0; k < bench_iterations; ++k)
    // for (int k = 0; k < 1; ++k)
    {
        vector<vector<int> > IN(n); // n is the rows
        vector<vector<int> > DA(n); // n is the rows
        vector<vector<int> > ptr(n, vector<int>(Ow + 1, 0)); // n is the rows
        clock_t t_cpo_creation_c;
        t_cpo_creation_c = clock();
        // Get the total number of non zeros: we can save it while encoding:
        CPO(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr);

        double elapsed = 1000 * ((double)(clock() - t_cpo_creation_c)) / CLOCKS_PER_SEC; // time in milliseconds
        if (k>0)
            t_cpo_creation_V7 +=  elapsed / (Ih * Iw * 1.0); // normalized timing
            // t_cpo_creation_V7 +=  elapsed ; // normalized timing
        if (k<1)
        {
            // cout << "\nOld 2D Ptr:" << endl;
            // print2DVector(ptr);
            // cout << "\n\nOld 2D IN:" << endl;
            // print2DVector(IN);

            // cout << "\n\nOld 2D DA:" << endl;
            // print2DVector(DA);


        }
        if (k != bench_iterations - 1)
        {
            IN.clear();
            DA.clear();
            reset_ptr(ptr_1d);
        } // end if
    }
    cout << "\nCPO_org time : "<< t_cpo_creation_V7 <<endl; // normalized timing 


    t_cpo_creation_V7 = 0   ;
    cout << "\n ####### CPO3_ wo rep ####### " << endl;
    for (int k = 0; k < bench_iterations; ++k)
    // for (int k = 0; k < 1; ++k)
    {
        vector<vector<int> > IN(n); // n is the rows
        vector<vector<int> > DA(n); // n is the rows
        vector<vector<int> > ptr(n, vector<int>(Ow + 1, 0)); // n is the rows
        clock_t t_cpo_creation_c;
        t_cpo_creation_c = clock();

        // the only thing done here is using of 1d pointer instead of 2d
        
        int count_d = CPO3_(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr_1d);
        double elapsed = 1000 * ((double)(clock() - t_cpo_creation_c)) / CLOCKS_PER_SEC; // time in milliseconds
        
        if (k>0)
            t_cpo_creation_V7 +=  elapsed / (Ih * Iw * 1.0); // normalized timing
            // t_cpo_creation_V7 +=  elapsed ; // normalized timing
        if (k<1)
        {
            // cout <<"\n\nNumber of Non-zero elements : " << count_d <<endl;
            cout << "\nNew 1D Ptr:" << endl;
            printVector(ptr_1d);

            // cout << "\nNew 2D DA:" << endl;
            // print2DVector(DA);

            cout << "\nNew 2D Index:" << endl;
            print2DVector(IN);
        // cout << "\nExiting Encoding V7" << endl;
        }
        if (k != bench_iterations - 1)
        {
            IN.clear();
            DA.clear();
            reset_ptr(ptr_1d);
        } // end if
    }
    cout << "\nCPO3_1D time : "<< t_cpo_creation_V7 <<endl; // normalized timing 

    t_cpo_creation_V7 = 0   ;
    cout << "\n ####### CPO3__ #######  " << endl;
    for (int k = 0; k < bench_iterations; ++k)
    // for (int k = 0; k < 1; ++k)
    {
        vector<vector<int> > IN(n); // n is the rows
        vector<vector<int> > DA(n); // n is the rows
        vector<vector<int> > ptr(n, vector<int>(Ow + 1, 0)); // n is the rows
        vector<vector<int> > ptr2d(n, vector<int>(Ow + 1, 0)); // n is the rows
        clock_t t_cpo_creation_c;
        t_cpo_creation_c = clock();

        // the only thing done here is using of 1d pointer instead of 2d
        
        int count_d = CPO3__(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr_1d, ptr2d);
        double elapsed = 1000 * ((double)(clock() - t_cpo_creation_c)) / CLOCKS_PER_SEC; // time in milliseconds
        
        if (k>0)
            t_cpo_creation_V7 +=  elapsed / (Ih * Iw * 1.0); // normalized timing
            // t_cpo_creation_V7 +=  elapsed ; // normalized timing
        if (k<1)
        {
            // cout <<"\n\nNumber of Non-zero elements : " << count_d <<endl;
            // cout << "\nNew 1D Ptr:" << endl;
            // printVector(ptr_1d);

            // cout << "\nNew 2D DA:" << endl;
            // print2DVector(DA);
        // cout << "\nNew 2D Index:" << endl;
        // print2DVector(IN);
        // cout << "\nExiting Encoding V7" << endl;
        }
        if (k != bench_iterations - 1)
        {
            IN.clear();
            DA.clear();
            reset_ptr(ptr_1d);
        } // end if
        
    }
    cout << "\nCPO3__1D time : "<< t_cpo_creation_V7 <<endl; // normalized timing 
    reset_ptr(ptr_1d,-1);

    {
        int m3_pos = Iw - 2 * Kw + 2;
        t_cpo_creation_V7 = 0   ;
        cout << "\n ####### CPO3___ #######  " << endl;
        std::vector<int> x(n, 0);
        std::vector<int> m(n, m3_pos);
        vector<vector<int> > IN(n); // n is the rows
        vector<vector<int> > DA(n); // n is the rows
        for (int k = 0; k < bench_iterations; ++k)
        // for (int k = 0; k < 1; ++k)
        {
            clock_t t_cpo_creation_c;
            t_cpo_creation_c = clock();

            // the only thing done here is using of 1d pointer instead of 2d
            
            int count_d = CPO3___(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr_1d, x, m);
            IN_1d.reserve(count_d);
            DA_1d.reserve(count_d);
            transform2dTo1d(IN, DA, IN_1d, DA_1d);
            // transform2dTo1d(IN, DA, IN_1d, DA_1d);
            double elapsed = 1000 * ((double)(clock() - t_cpo_creation_c)) / CLOCKS_PER_SEC; // time in milliseconds
            
            if (k>0)
                t_cpo_creation_V7 +=  elapsed / (Ih * Iw * 1.0); // normalized timing
                // t_cpo_creation_V7 +=  elapsed ; // normalized timing
            if (k<1)
            {
                // cout <<"\n\nNumber of Non-zero elements : " << count_d <<endl;
                // cout << "\nNew 1D Ptr:" << endl;
                // printVector(ptr_1d);

                // cout << "\nNew 2D DA:" << endl;
                // // print2DVector(DA);
                // printVector(DA_1d);

                // cout << "\nNew 2D Index:" << endl;
                // print2DVector(IN);
                // printVector(IN_1d);
            // cout << "\nExiting Encoding V7" << endl;
            }

            if (k != bench_iterations - 1)
            {
                IN.clear();
                DA.clear();
                IN_1d.clear();
                DA_1d.clear();
                reset_ptr(ptr_1d,-1);
                reset_ptr(x,0);
                reset_ptr(m,m3_pos);
            } // end if

        }
        cout << "\nCPO3___ 1D time : "<< t_cpo_creation_V7 <<endl; // normalized timing 
    }

    reset_ptr(ptr_1d,0);
    {
        t_cpo_creation_V7 = 0   ;
        cout << "\n ####### CPO3 #######  " << endl;
        std::vector<int> x(n, 0);
        std::vector<int> m(n, 0);
        vector<vector<int> > IN(n); // n is the rows
        vector<vector<int> > DA(n); // n is the rows
        for (int k = 0; k < bench_iterations; ++k)
        // for (int k = 0; k < 1; ++k)
        {
            clock_t t_cpo_creation_c;
            t_cpo_creation_c = clock();

            // the only thing done here is using of 1d pointer instead of 2d
            
            int count_d = CPO3(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr_1d, x, m);
            transform2dTo1d(IN, DA, IN_1d, DA_1d);
            double elapsed = 1000 * ((double)(clock() - t_cpo_creation_c)) / CLOCKS_PER_SEC; // time in milliseconds
            
            if (k>0)
                t_cpo_creation_V7 +=  elapsed / (Ih * Iw * 1.0); // normalized timing
                // t_cpo_creation_V7 +=  elapsed ; // normalized timing
            if (k<1)
            {
                // cout <<"\n\nNumber of Non-zero elements : " << count_d <<endl;
                // cout << "\nNew 1D Ptr:" << endl;
                // printVector(ptr_1d);

                // cout << "\nNew 2D DA:" << endl;
                // print2DVector(DA);
                // cout << "\nNew 2D Index:" << endl;
                // print2DVector(IN);
            // cout << "\nExiting Encoding V7" << endl;
            }

            if (k != bench_iterations - 1)
            {
                IN.clear();
                DA.clear();
                reset_ptr(ptr_1d);
                reset_ptr(x,0);
                reset_ptr(m,0);
            } // end if
        }
        cout << "\nCPO3 1D time : "<< t_cpo_creation_V7 <<endl; // normalized timing 

    }
    
    reset_ptr(ptr_1d,-1);

    {
        int m3_pos = Iw - 2 * Kw + 2;
        t_cpo_creation_V7 = 0   ;
        cout << "\n ####### CPO5_OP #######  " << endl;
        std::vector<int> x(n, 0);
        std::vector<int> m(n, m3_pos);
        for (int k = 0; k < bench_iterations; ++k)
        // for (int k = 0; k < 1; ++k)
        {
            clock_t t_cpo_creation_c;
            t_cpo_creation_c = clock();
            // the only thing done here is using of 1d pointer instead of 2d
            // CPO5_OP(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, ptr_1d, IN_1d, DA_1d, x, m);
            CPO_EncodingV8(IN_1d, DA_1d, ptr_1d, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, m, n);

            double elapsed = 1000 * ((double)(clock() - t_cpo_creation_c)) / CLOCKS_PER_SEC; // time in milliseconds
            
            if (k>0)
                t_cpo_creation_V7 +=  elapsed / (Ih * Iw * 1.0); // normalized timing
                // t_cpo_creation_V7 +=  elapsed ; // normalized timing

            if (k == bench_iterations - 1)
            {
                // cout <<"\n\nNumber of Non-zero elements : " << count_d <<endl;
                cout << "\nNew 1D Ptr:" << endl;
                printVector(ptr_1d);

                // cout << "\nNew 2D DA:" << endl;
                // // print2DVector(DA);
                // printVector(DA_1d);

                cout << "\nNew 2D Index:" << endl;
                printVector(IN_1d);
                // printVector(IN_1d);
            // cout << "\nExiting Encoding V7" << endl;
            }
            if (k != bench_iterations - 1)
            {
                IN_1d.clear();
                DA_1d.clear();
                reset_ptr(ptr_1d,-1);
                reset_ptr(m,m3_pos);
            } // end if

        }
        cout << "\nCPO5 1D time : "<< t_cpo_creation_V7 <<endl; // normalized timing 
    }


    {
        // CSR Encoding:
        cout << "\n ####### CSCC #######  " << endl;
        std::vector<int> DA_1d_csr;
        std::vector<int> IN_1d_csr;
        std::vector<int> ptr_1d_csr(Ow+1,0);
        for (int k = 0; k < bench_iterations; ++k)
        {
            // copy data to double

            clock_t t_cpo_creation_c;
            t_cpo_creation_c = clock();
            CSR(DA_1d_csr, IN_1d_csr, ptr_1d_csr, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
            // CSR2(DA_1d_v7_d, IN_1d_v7_d, ptr_1d_v7, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
            double elapsed = 1000 * ((double)(clock() - t_cpo_creation_c)) / CLOCKS_PER_SEC; // time in milliseconds
            if (k > 0)
                t_cscc_creation += elapsed / (Ih * Iw * 1.0); // normalized timing

            if (k<1)
            {
                // cout <<"\n\nNumber of Non-zero elements : " << count_d <<endl;
                // cout << "\nNew 1D Ptr:" << endl;
                // printVector(ptr_1d_csr);

                // cout << "\nNew 2D DA:" << endl;
                // print2DVector(DA);
                // cout << "\nNew 2D Index:" << endl;
                // printVector(IN_1d_csr);
            // cout << "\nExiting sEncoding V7" << endl;
            }

            if (k != bench_iterations - 1)
            {
                IN_1d_csr.clear();
                DA_1d_csr.clear();
                ptr_1d_csr.clear();
            } // end if

        } // end for
        cout << "\n\nCSCC encoding time : " << t_cscc_creation << endl << endl;
    }
    // exit(0);


    // transform 2d to 1d:
    // int count_ptr = n*(1+Ow);
    // if(n != 1)
    // {
    //   count_ptr = (n-1)*(1 + Ow) + min(int(ptr[0].size()), 3);
    // }

    // IN_1d.reserve(count_d);
    // DA_1d.reserve(count_d);
    // // ptr_1d.reserve(count_ptr);  

    // transform2dTo1d(IN, DA, IN_1d, DA_1d);

    // if(n != 1)
    // {
    //   transform2dTo1dV7(IN, DA, ptr, IN_1d, DA_1d, ptr_1d);  
    // }
    // else
    // {
    //   transform2dTo1d_old(IN, DA, ptr, IN_1d, DA_1d, ptr_1d);
    // }

    cout << " *********************************************************************" << endl;

}

int CPO3(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> > &IN,
         vector<vector<int> > &DA, vector<int> &ptr)
{


  int l = 0;
  int n = Kw;
  int non_zero_count = 0;


  std::vector<int> x(n, 0);
  std::vector<int> m(n, 0);

    // First part
    for (int j = 0; j < Kw; ++j)
    {
        // ptr[l][m[l]] = x[l];
        ptr[l*(1 + Ow) +  m[l]] = x[l];
        m[l]++;
        for (int i = 0; i < Ih; ++i)
        {
            if (org_mat(i, j) != 0)
            {
                int ind_val = j + (i * Kw);
                IN[l].push_back(ind_val);
                DA[l].push_back(org_mat(i, j));
                x[l]++;
                non_zero_count++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)       
        // ptr[l][m[l]] = x[l];
        ptr[l*(1 + Ow) +  m[l]] = x[l];
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
                non_zero_count++;
            } // end if (org_mat(i, j) != 0)
        }// end for(i = 0; i < Ih; ++i)
        // ptr[l][m[l]] = x[l];
        ptr[l*(1 + Ow) +  m[l]] = x[l];
        m[l]++;
        if (n > 1)
        {
            for (int c = 1; c < n - 1; ++c)
            // for (int c = 1; c < 4; ++c)
            {
                // ptr[c][m[c]] = x[c];
                ptr[c*(1 + Ow) +  m[c]] = x[c];
                m[c]++;
                non_zero_count++;
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
                if (j != Iw - 1)
                {
                    int ind_val = j - (m[l] - 1) + (i * Kw);
                    IN[l].push_back(ind_val);
                    DA[l].push_back(org_mat(i, j));
                    x[l]++;
                }
                else {
                    int ind_val = Kw - 1 + (i * Kw);
                    IN[l].push_back(ind_val);
                    DA[l].push_back(org_mat(i, j));
                    x[l]++;
                }

                non_zero_count++;

            }// end if(org_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)
        for (int c = 0; c < l + 1; ++c)
        {
            // ptr[l][m[l]] = x[l];
            ptr[l*(1 + Ow) +  m[l]] = x[l];
            m[l]++;
        } // end for(int c = 0; c < l+1; ++c)
        if (l > 1)
        {
            for (int c = 1; c < l; ++c)
            {
                // ptr[c][m[c]] = x[c];
                ptr[c*(1 + Ow) +  m[c]] = x[c];
                m[c]++;
            } // end for(int c = 0; c < l-1; ++c)
        }// end if l > 1
        l--;
    } // end for (j = Iw - Kw; Iw; ++j)

   // std::cout << "\nPtr: ";
  // print2DVector(ptr);

   // std::cout << "\n\nIN: ";
  // print2DVector(IN);

   //  std::cout << "\n\nData:";
    //  print2DVector(DA);
   // std::cout << "\n" << endl;

  return non_zero_count;
}



int CPO2(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> > &IN,
         vector<vector<int> > &DA, vector<int> &ptr)
{

  int flag = 0;
  int i = 0;
  int l = 0;
  int n = ceil(Kw / Sw);

  int non_zero_count = 0;

  std::vector<int> x(n, 0);
  std::vector<int> m(n, 0);

    // First piece
    for (int j = 0; j < Kw; ++j)
    {
        if (flag == 0)
        {
            // ptr[l][m[l]] = x[l];
            ptr[l*(1 + Ow) +  m[l]] = x[l];
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
                non_zero_count++;
            } // end if  if (org_mat(i, j) != 0)
        } // end for(i=0; i < Ih; ++i)
        if ((j + 1) % Sw == 0)
        {
            // ptr[l][m[l]] = x[l];
            ptr[l*(Ow + 1) +  m[l]] = x[l];
            l++;
            flag = 0;
        } // end if ( (j+1) % Sw == 0)
        else if (j == Kw - 1)
        {
            // ptr[l][m[l]] = x[l];
            ptr[l*(Ow + 1) +  m[l]] = x[l];
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
                non_zero_count++;
            } // end if (org_mat(i, j) != 0)
        }// end for(i = 0; i < Ih; ++i)

        if (flag == 0)
        {
            if (Kw % Sw == 0)
            {
                if ((j - Kw + 1) % Sw == 0)
                {
                    // ptr[l][m[l]] = x[l];
                    ptr[l*(Ow + 1) +  m[l]] = x[l];
                    m[l]++;

                    if (n > 1)
                    {
                        for (int c = 0; c < n - 1; ++c)
                        {
                            // ptr[c][m[c]] = x[c];
                            ptr[c*(Ow + 1) +  m[c]] = x[c];
                            m[c]++;
                        } // end for(int c = 0; c < n-1; ++c)
                    } // end if(n > 1)
                } // end if ((j - Kw + 1) % Sw == 0)
            } // end if (Kw % Sw == 0)
            else if ((j - Kw + 1) % Sw == (Sw - (Kw % Sw)))
            {
                // ptr[l][m[l]] = x[l];
                ptr[l*(Ow + 1) +  m[l]] = x[l];
                m[l]++;
                l++;
                flag = 1;
            } // end else if ( (j - Kw + 1) % Sw == (Sw - (Kw % Sw)))
        } // end if(flag == 0)
        else
        {
            if ((j - Kw + 1) % Sw == 0)
            {
                // ptr[l][m[l]] = x[l];
                ptr[l*(Ow + 1) +  m[l]] = x[l];
                m[l]++;
                l--;
                flag = 0;
            } // end if ( (j - Kw + 1) % Sw == 0)
            if (n > 2)
            {
                for (int c = 0; c < n - 2; ++c)
                {
                    // ptr[c][m[c]] = x[c];
                    ptr[c*(Ow + 1) + m[c]] = x[c];
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
                non_zero_count++;
            }// end if(lowered_mat(i, j) != 0)
        } // end for(i = 0; i = Ih; ++i)
        if ((Iw - j - 1) % Sw == 0)
        {
            for (int c = 0; c < l + 1; ++c)
            {
                // ptr[l][m[l]] = x[l];
                ptr[l*(Ow + 1) +  m[l]] = x[l];
                m[l]++;
            } // end for(int c = 0; c < l+1; ++c)
            if (l > 1)
            {
                for (int c = 0; c < l; ++c)
                {
                    // ptr[c][m[c]] = x[c];
                    ptr[c*(Ow + 1) + m[c]] = x[c];
                    m[c]++;
                } // end for(int c = 0; c < l-1; ++c)
            }// end if l > 1

            else if (l == 1)
            {
                // ptr[0][m[0]] = x[0];
                ptr[0*(Ow + 1) +  m[0]] = x[0];
                m[0]++;
            } // end else if(l == 1)
            l--;
        } // end if ((Iw - j - 1) % Sw == 0)
    } // end for (j = Iw - Kw; Iw; ++j)

  // std::cout << "\nPtr: ";
  // printVector(ptr);
  // exit(0);

  // std::cout << "\n\nIN: ";
  // print2DVector(IN);

  // std::cout << "\n\nData:";
  // print2DVector(DA);
  // std::cout << "\n" << endl;
  return non_zero_count;
}


int CPO1(MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> > &IN,
         vector<vector<int> > &DA, vector<vector<int> > &ptr)
{

  int flag = 0;
  int i = 0;
  int l = 0;
  int n = ceil(Kw / Sw);

  int non_zero_count = 0;

  std::vector<int> x(n, 0);
  std::vector<int> m(n, 0);

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
                non_zero_count++;
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
                non_zero_count++;
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
                non_zero_count++;
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
  // print2DVector(ptr);

  // std::cout << "\n\nIN: ";
  // print2DVector(IN);

  // std::cout << "\n\nData:";
  // print2DVector(DA);
  // std::cout << "\n" << endl;
  return non_zero_count;
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


void conv_CPO_v3(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<int> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;
    
    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int *x_ptr           = &ptr[0];
    int  x               = *x_ptr;
    int *Aindex_help     = &IN[x];
    int *Adata_help      = &DA[x];

    // For each ptr type
    for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    // int type_ptr = 0;
    {

     
    // For each submat
    for (int submat = 0; submat < number; ++submat)
    {
     
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

      // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;

        
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        int used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
      
        for(int i = 0; i <= type_ptr; ++i)
        {

          // Loop on Kh for the output
        for(int l = 0; l < Kh; ++l)
        {
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int input_index  = used_index - i;
            int y_out        = (input_index/Kw) - l;
            int x_out        = i + submat;
        
           if(y_out < 0 || y_out >= Oh){
            // cout << "continue YYY============\n" << endl;
            continue;
         }
        
    
        // cout << "R) " << y_out << ", C) " << x_out << ", Data: " << DA[type_ptr][x] << ", Index: " << input_index  << ", ac_Index: " << IN[type_ptr][x] << endl;
       //    // O(y_out, x_out) += DA[type_ptr][x] * 1.0;
          O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
      
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


/*SSSSSSSSSSSSSSSSSSSSSSSSSSS*/

// Final Implementation by Ahmad: September 30, 2020
void CSR(std::vector<int>& DA, std::vector<int>& IN, std::vector<int>& ptr, MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
    int i = 0;
    int x = 0; // 
    int m = 0; //submatrix
    int ptr_count = 1;
    // ptr.push_back(x);
    ptr[0] = 0;
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
                ptr[ptr_count] = x;
                ptr_count++;
                if (j == Iw - Kw)
                {
                    break;
                }
            } // end if (i == Ih)
        } // end if (j == m + (Kw - 1))
    } // end for (int j = 0; j < Iw; ++j)
    // cout << "\nCSR Done\n" << endl;
   // std::cout << "\nPtr: ";
   // printVector(ptr);

  //  std::cout << "\n\nIN: ";
  //  printVector(IN);

   // std::cout << "\n\nData:";
   // printVector(DA);
  //  std::cout << "\n" << endl;
}

// CSR without eigen and creation of ptr
// void CSR2(MatrixXf& O, VectorXf& K, MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
void CSR2(std::vector<double> &DA, std::vector<int> &IN, std::vector<int> &ptr, MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
    int x = 0;
    int m = 0;

  
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

    ptr.push_back(x);
    //std::cout << "\nPtr: ";
    //printVector(ptr);

   // std::cout << "\n\nIN: ";
   // printVector(IN);

   // std::cout << "\n\nData:";
   // printVector(DA);
   // std::cout << "\n" << endl;
}

void CSR1(std::vector<double> &DA, std::vector<int> &IN, std::vector<int> &ptr, MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
    int i = 0;
    int x = 0;
    int m = 0;


    std::cout << "\n===Original Feature Map (" << Ih << "x" << Iw <<  "):  \n" << org_mat << std::endl;
    cout << "-----\n" << endl;

    ptr.push_back(x);
    for (int j = 0; j < Iw; ++j)
    {
        if (org_mat(i, j) != 0)
        {
            int index = j + (i * Kw) - m;

            IN.push_back(index);
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

   // std::cout << "\nPtr: ";
  //  printVector(ptr);

  //  std::cout << "\n\nIN: ";
   // printVector(IN);

    //std::cout << "\n\nData:";
   //printVector(DA);
  // std::cout << "\n" << endl;
}


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
        // if(m < 0 || m >= Oh) continue;
                 
        // cout << "R) " << m << ", C) " << n << ", " << Kindex << endl;
        if(m >= 0 && m < Oh)
          O[m][n] += NZE_data*K[Kindex];
      }   
    }   

  }

    
} // end mult

void csrMult_v5(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<double> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    // printVector(ptr);
    // exit(0);

    int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr

    // number = 0; n = 1;
    
    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int *x_ptr           = &ptr[0];
    int  x               = *x_ptr;
    int *Aindex_help     = &IN[x];
    double *Adata_help   = &DA[x];

    // For each ptr type
    // for (int type_ptr = 0; type_ptr < n; ++type_ptr)
 
   // For each submat
    for (int submat = 0; submat < number; ++submat)
    {

      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

      // cout << "Sumbat: " << submat << ", type_ptr: " << 0 << " start " << x  << " end: " << end_x_loop  << endl;

   
      for(; x < end_x_loop; ++x)
      {    

        
       // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  
        // cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << end_x_loop << " ind to fetch: " <<  type_ptr*number + submat + 1 << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        int used_data   = *Adata_help;  Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
        for(int l = 0; l < Kh; ++l)
        {
              // I = 0:
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int input_index  = used_index;
            int y_out        = (input_index/Kw) - l;
            int x_out        = submat;

            if(y_out >= 0 && y_out < Oh) {
               O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
          } // end if(y_out >= 0 && y_out < Oh) 

       } // end l loop

      } // end x
    } // end sumbat

    

    // cout << "Output: " << endl;
    // print2DVectorF(O);
    // cout << "-----\n" << endl;
}


int main()
{
    
    
    // float density = 0.1;
//    float density = 0.5;

  // bench iterations
     int bench_iterations = 100000;
    // int bench_iterations = 10000;
    // int bench_iterations = 1;
  //int bench_iterations = 1;

  // std::vector<int> I_list = {8,17,50};
  // std::vector<int> Kh_list = {3, 7, 1};
  // std::vector<int> Kw_list = {3,1, 7};


  // Big test
   std::vector<int> I_list = {50, 8, 17};
   std::vector<int> Kh_list = {3, 1, 3, 7, 1};
   std::vector<int> Kw_list = {3, 3, 1, 1, 7};

    //std::vector<int> Kh_list = { 3, 3 };
    //std::vector<int> Kw_list = { 3, 1 };


  // std::vector<int> I_list = { 8,17,50 };
  // std::vector<int> Kh_list = { 3, 7, 1 };
  // std::vector<int> Kw_list = { 1, 1, 7 };

   // std::vector<int> I_list = { 50 };
   // std::vector<int> Kh_list = { 3};
   // std::vector<int> Kw_list = { 1};

   //std::vector<int> I_list = {17};
   //std::vector<int> Kh_list = {1};
   //std::vector<int> Kw_list = {5};

  // std::vector<int> I_list = {17};
  // std::vector<int> Kh_list = {5};
  // std::vector<int> Kw_list = {5};


  // std::vector<int> I_list = {50};
  // std::vector<int> Kh_list = {3};
  // std::vector<int> Kw_list = {3};

  // std::vector<int> I_list = {8};
  // std::vector<int> Kh_list = {3, 3};
  // std::vector<int> Kw_list = {3, 1};


  // std::vector<int> I_list = {17};
  // std::vector<int> Kh_list = {1};
  // std::vector<int> Kw_list = {7};
    
    for(int KK = 0; KK < Kh_list.size(); ++KK)
    {
    
      // for(int I: I_list)
      for(int II = 0; II < I_list.size(); ++II)
      {
        int Ih = I_list[II];
        int Iw = I_list[II];

        // density:e
     // float density = 0.1;
        float density = 0.05;
    //    float density = 0.3;
  // float density = 1;
    

    for(; density < 1.05; density+=0.05)
    {
        
        // timer for im2col, csr
        float t_im2col = 0;
        float t_csr    = 0;
        float t_cpoV5    = 0;
        float t_cpoV6    = 0;
        float t_cpoV7    = 0;
        float t_cpoV8    = 0;

        // timer for creations:
        float t_cpo_creation = 0;
        float t_cscc_creation = 0;
        float t_im2col_creation = 0;
        float t_cpo_creation_woRep = 0;
        float t_cpo_creation_V7 = 0;
        float t_cpo_creation_V8 = 0;

        // space for im2col, csr
        float s_im2col = 0;
        float s_csr    = 0;
        float s_cpo    = 0;
        
    
        
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
        if((Kh == 1 && Kw == 7) || Kh == 7)
          continue;


        
        
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
        double density_cal = generate_org_featureMap(org_fm, Ih, Iw, density);
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
        
        
        // Create the intermediate representation for im2col:
        MatrixXf im2col_mat    = MatrixXf::Zero(Oh*Ow, Kh*Kw*Ic);
        for(int k = 0; k < bench_iterations; ++k)
        {

          clock_t t_im2col_creation_c;
          t_im2col_creation_c = clock();
          Im2col_Encoding(im2col_mat, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, Ic);
          double elapsed  = 1000*((double)(clock()-t_im2col_creation_c))/CLOCKS_PER_SEC; // time in milliseconds
          if(k > 0)
            t_im2col_creation += elapsed/(Ih*Iw*1.0); // normalized timing

          // Reset im2col encoing
          if(k != bench_iterations-1)
          {
            reset_Im2col_Encoding(im2col_mat);  
          }
        }
#if case_1_time
        cout << "\t\t\tIm2col Encoding" << endl;
        cout << t_im2col_creation << endl;
#endif
        // Space for im2col:
        s_im2col = im2col_mat.rows()*im2col_mat.cols();

        // std::cout << "\n===Im2col Feature Map of Size: " << im2col_mat.rows() << ", " << im2col_mat.cols() <<  "\n" << im2col_mat << std::endl;
        // cout << "-----\n" << endl;
        
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

        // Prepare the output for CPO
        vector<vector<float> > O( Oh , vector<float> (Ow, 0));
        
        // Create the Kernel
        vector<int> Kernel(Kh*Kw, 1);
        
        // transpose the matrix for im2col:
        MatrixXf im2col_mat_tr = im2col_mat.transpose();
        
        // Perform 50 times dense matrix dense vector multiplication: d_o1 = d_m * d_b
        {
            // clock_t t;
            // t = clock();
            // for(int k=0;k<bench_iterations;k++)  bench_Dense(im2col_mat, filter_vectorized, d_o1);
            // double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
            // t_im2col = elapsed/(Ih*Iw*1.0); // normalized timing

            for(int k=0;k<bench_iterations;k++){

                clock_t t;
                t = clock();
                bench_Dense(im2col_mat, filter_vectorized, d_o1);

                double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds

                 if (k > 0)
                  t_im2col += elapsed/(Ih*Iw*1.0); // normalized timing
            } 
            
            // include creation time:
            t_im2col +=  t_im2col_creation;
#if case_1_time
        cout << "\t\t\tIm2col ALL" << endl;
        cout << t_im2col << endl;
#endif
        }
        
        // number of ptrs:
        int n = ceil(Kw / Sw);
        if (Kw % Sw == 0)
        {
         n = Kw / Sw;
        }

        // CPO Encoding:
        std::vector<int> IN_1d;
        std::vector<int> DA_1d;

        // transform 2d to 1d:
        int count_ptr = n*(1+Ow);
        std::vector<int> ptr_1d(count_ptr, 0);
        int count_d;
    { 
        for(int k = 0; k < bench_iterations; ++k)
        {

          clock_t t_cpo_creation_c;
          t_cpo_creation_c = clock();
          count_d = CPO_Encoding(IN_1d, DA_1d, ptr_1d, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
          double elapsed  = 1000*((double)(clock()-t_cpo_creation_c))/CLOCKS_PER_SEC; // time in milliseconds
          if(k > 0)
            t_cpo_creation += elapsed/(Ih*Iw*1.0); // normalized timing

          if (k<1)
          {
            // cout << "CPO ptr : " << endl;
            // printVector(ptr_1d);
            // cout << "CPO IN : " << endl;
            // printVector(IN_1d);
          }

          if(k != bench_iterations-1)
          {
            IN_1d.clear();
            DA_1d.clear();
            // ptr_1d.clear();  
            reset_ptr(ptr_1d);
          }
          
        }
#if case_1_time
        cout << "\t\t\tCPO_org Encoding" << endl;
        cout << t_cpo_creation << endl;
#endif
#if case_1
        cout << "Ptr: " << endl;
        printVector(ptr_1d);

        cout << "Index: " << endl;
        printVector(IN_1d);
#endif
        // cout << "Data:  " << endl;
        // printVector(DA_1d);


        // cout << "\nPtr2: ";
        // print2DVector(ptr);
    
        // cout << "\n\nIN: ";
        // print2DVector(IN);
    
        // cout << "\n\nData:";
        // print2DVector(DA);
        // cout << "\n" << endl;
        // cout << t_cpo_creation << endl;
    }
        ///////////

        // V3 Code:
         // Perform 50 times raw sparse matrix dense vector multiplication: d_CPO = d_m * d_b
            // Prepare the output for CPO
        vector<vector<float> > O_CPO2( Oh , vector<float> (Ow, 0));
        {
            for(int k=0;k<bench_iterations;k++)
            {
                clock_t t4;
                t4 = clock();
                // conv_CPO_v4(O_CPO2, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                // Conv CPO V3: 
                conv_CPO_v3(O_CPO2, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                // conv_CPO_v5(O_CPO2, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

              // Hossam: call if n = 3
              // if(n == 3)
              //   conv_CPO_v6(O_CPO2, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

               // if(k == bench_iterations-1)
               // {
               //     print2DVectorF(O_CPO2);
               //     cout << "-----\n" << endl;
               // }

                double elapsed  = 1000*((double)(clock()-t4))/CLOCKS_PER_SEC; // time in milliseconds

                if(k > 0)
                t_cpoV6 += elapsed/(Ih*Iw*1.0); // normalized timing
            } // end k lop 
                // include creation time:
               t_cpoV6 += t_cpo_creation;
        }     
#if case_1_time
        cout << "\t\t\tCPO_org All" << endl;
        cout << t_cpoV6 << endl;
#endif


#if encode_all
        All_CPO(IN_1d_v7, DA_1d_v7, ptr_1d_v7, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
#endif

#if 0
        {
            int count_ptr_v8 = n*(1+Ow);
            if(n != 1)
            {
              count_ptr_v8 = (n-1)*(1 + Ow) + min(1 + Ow, 3);
            }
            t_cpo_creation_woRep = 0   ;
            

            vector<vector<int> > IN(n); // n is the rows
            vector<vector<int> > DA(n); // n is the rows
            vector<int> ptr_1d(count_ptr_v8,-1); // n is the rows
            // vector<vector<int> > ptr2d(n, vector<int>(Ow + 1, 0)); // n is the rows

            for (int k = 0; k < bench_iterations; ++k)
            // for (int k = 0; k < 1; ++k)
            {
                clock_t t_cpo_creation_c;
                t_cpo_creation_c = clock();

                // the only thing done here is using of 1d pointer instead of 2d
                
                int count_d = CPO3_(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr_1d);
                double elapsed = 1000 * ((double)(clock() - t_cpo_creation_c)) / CLOCKS_PER_SEC; // time in milliseconds
                
                if (k>0)
                    t_cpo_creation_woRep +=  elapsed / (Ih * Iw * 1.0); // normalized timing
                    // t_cpo_creation_V7 +=  elapsed ; // normalized timing
                if (k<1)
                {
                    // cout <<"\n\nNumber of Non-zero elements : " << count_d <<endl;
                    // cout << "\nNew 1D Ptr:" << endl;
                    // printVector(ptr_1d);

                    // cout << "\nNew 2D DA:" << endl;
                    // print2DVector(DA);

                    // cout << "\nNew 2D Index:" << endl;
                    // print2DVector(IN);
                // cout << "\nExiting Encoding V7" << endl;
                }
                if (k != bench_iterations - 1)
                {
                    IN.clear();
                    DA.clear();
                    reset_ptr(ptr_1d);
                } // end if
            }
        }

#endif

#if case_1_time
        cout << "\n\t\t\tCPO3_ wo rep" << endl;
        cout << t_cpo_creation_woRep <<endl; // normalized timing
#endif

#if case_1
            cout << "New Ptr: " << endl;
            printVector(ptr_1d);

            cout << "New Index: " << endl;
            print2DVector(IN);
            // exit(0);
#endif
        // V7 Code:
#if 1
        // CPO Encoding V7:
        // int count_d = ceil(density*Iw*Ih);
        std::vector<int> IN_1d_v7;
        std::vector<int> DA_1d_v7;

        int count_ptr_v8 = n*(1+Ow);
        if(n != 1)
        {
          count_ptr_v8 = (n-1)*(1 + Ow) + min(1 + Ow, 3);
        }
        std::vector<int> ptr_1d_v7(count_ptr_v8, -1);
        // transform2dTo1d(IN, DA, IN_1d, DA_1d);

        {
            int m3_pos = Iw - 2 * Kw + 2;
            std::vector<int> x(n, 0);
            std::vector<int> m(n, m3_pos);
            vector<vector<int> > IN_v7(n); // n is the rows
            vector<vector<int> > DA_v7(n); // n is the rows
            for(int k = 0; k < bench_iterations; ++k)
            {

              clock_t t_cpo_creation_c;
              t_cpo_creation_c = clock();
              CPO_EncodingV7(IN_1d_v7, DA_1d_v7, ptr_1d_v7, org_fm, Kh, Kw, Oh, Ow, 
                                    Sh, Sw, Ih, Iw, IN_v7, DA_v7, x, m, n);
              double elapsed  = 1000*((double)(clock()-t_cpo_creation_c))/CLOCKS_PER_SEC; // time in milliseconds
              if(k > 0)
                t_cpo_creation_V7 += elapsed/(Ih*Iw*1.0); // normalized timing

              if(k != bench_iterations-1)
              {
                IN_1d_v7.clear();
                DA_1d_v7.clear();
                IN_v7.clear();
                DA_v7.clear();
                IN_v7.resize(n);
                DA_v7.resize(n);

                // ptr_1d_v7.clear(); 
                reset_ptr(ptr_1d_v7,-1);
                reset_ptr(x,0);
                reset_ptr(m,m3_pos);
              }
              
            }
#if case_1_time
            cout << "\t\t\tCPO_EncodingV7" << endl;
            cout << t_cpo_creation_V7 << endl;
#endif

#if case_1
            cout << "New Ptr: " << endl;
            printVector(ptr_1d_v7);

            cout << "Data: " << endl;
            printVector(DA_1d_v7);

            cout << "New Index: " << endl;
            printVector(IN_1d_v7);
            // exit(0);
#endif

            vector<vector<float> > O_v7( Oh , vector<float> (Ow, 0));
            
            if(n != 1)
            {
                
                // copy data to double
                std::vector<double> DA_1d_v7_d(DA_1d_v7.size(), 0);

                for(int h = 0; h < DA_1d_v7.size(); ++h)
                {
                  DA_1d_v7_d[h] = DA_1d_v7[h];
                }

                for(int k=0;k<bench_iterations;k++)
                {
                  clock_t t4;
                  t4 = clock();
                  conv_CPO_v8_trim(O_v7, Kernel, IN_1d_v7,  DA_1d_v7_d, ptr_1d_v7, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                 // if(k == bench_iterations-1)
                 // {
                 //     print2DVectorF(O_v7);
                 //     cout << "-----\n" << endl;
                 // }

                 double elapsed  = 1000*((double)(clock()-t4))/CLOCKS_PER_SEC; // time in milliseconds

                 if(k > 0)
                 t_cpoV7 += elapsed/(Ih*Iw*1.0); // normalized timing
              } // end k loop
      
              // Space
              s_cpo = IN_1d_v7.size() + DA_1d_v7_d.size() + ptr_1d_v7.size();

            } // end if
            else
            {

                // copy data to double
                std::vector<double> DA_1d_v7_d(DA_1d.size(), 0);

                for(int h = 0; h < DA_1d.size(); ++h)
                {
                  DA_1d_v7_d[h] = DA_1d[h];
                }


            for(int k=0;k<bench_iterations;k++){


            clock_t t5;
            t5 = clock();
                // conv_CPO_v7(O_v7, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
              conv_CPO_v7(O_v7, Kernel, IN_1d,  DA_1d_v7_d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

              double elapsed  = 1000*((double)(clock()-t5))/CLOCKS_PER_SEC; // time in milliseconds

              if(k > 0)
                t_cpoV7 += elapsed/(Ih*Iw*1.0); // normalized timing
              // cout << "a2: " << t_cpoV8 << " " << elapsed << endl;
            
            } // end k lop 
                    
            // Space
            s_cpo = IN_1d.size() + DA_1d_v7_d.size() + ptr_1d.size();

            } // end else

              // Include the creation time:
              t_cpoV7 += t_cpo_creation_V7;
        }

#if case_1_time
        cout << "\t\t\tCPO v7" << endl;
        cout << t_cpoV7 << endl;
#endif

#endif
            std::vector<int> ptr_1d_v8(count_ptr_v8, -1);
            std::vector<int> IN_1d_v8;
            std::vector<int> DA_1d_v8;
            int m3_pos = Iw - 2 * Kw + 2;
            std::vector<int> m(n, m3_pos);
            {
                for (int k = 0; k < bench_iterations; ++k)
                // for (int k = 0; k < 1; ++k)
                {
                    clock_t t_cpo_creation_c;
                    t_cpo_creation_c = clock();
                    // the only thing done here is using of 1d pointer instead of 2d
                    // CPO5_OP(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, ptr_1d_v8, IN_1d_v8, DA_1d_v8, x, m);
                    CPO_EncodingV8(IN_1d_v8, DA_1d_v8, ptr_1d_v8, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, m, n);
                    double elapsed = 1000 * ((double)(clock() - t_cpo_creation_c)) / CLOCKS_PER_SEC; // time in milliseconds
                    if (k>0)
                        t_cpo_creation_V8 +=  elapsed / (Ih * Iw * 1.0); // normalized timing
                        // t_cpo_creation_V7 +=  elapsed ; // normalized timing
                    // if (k == bench_iterations - 1)
                    // {
                        // for (int i = 0 ; i < ptr_1d_v8.size(); i++)
                        // {
                        //     if (ptr_1d_v8[i] < 0) ptr_1d_v8[i] = ptr_1d_v8[i-1];
                        // }
                        // cout << "\nNew 1D Ptr:" << endl;
                        // printVector(ptr_1d_v8);

                        // cout << "\nNew 1D Index:" << endl;
                        // printVector(IN_1d_v8);

                        // cout << "\nNew 2D DA:" << endl;
                        // // print2DVector(DA);
                        // printVector(DA_1d);

                        // printVector(IN_1d);
                    // cout << "\nExiting Encoding V7" << endl;
                    // }
                    if (k != bench_iterations - 1)
                    {
                        IN_1d_v8.clear();
                        DA_1d_v8.clear();
                        reset_ptr(ptr_1d,-1);
                        reset_ptr(m,m3_pos);
                    } // end if

                }
            }

#if case_1_time
            cout << "\t\t\tCPO_EncodingV8" << endl;
            cout << t_cpo_creation_V8 << endl;
#endif
            // cout << "Ptr: " << endl;
            // printVector(ptr_1d);
        // }

        // exit(0);
        // CPO V8:
#if case_1
            cout << "New Ptr: " << endl;
            printVector(ptr_1d_v8);

            // cout << "Data: " << endl;
            // printVector(DA_1d_v8);

            cout << "New Index: " << endl;
            printVector(IN_1d_v8);
            exit(0);
 #endif      
        // {
        // Prepare the output for CPO
        vector<vector<float> > O_v8( Oh , vector<float> (Ow, 0));
        {  
            if(n != 1)
            {
                
                // copy data to double
                std::vector<double> DA_1d_v8_d(DA_1d_v7.size(), 0);

                for(int h = 0; h < DA_1d_v8.size(); ++h)
                {
                  DA_1d_v8_d[h] = DA_1d_v8[h];
                }

                for(int k=0;k<bench_iterations;k++)
                {
                  clock_t t4;
                  t4 = clock();
                  conv_CPO_v8_trim(O_v8, Kernel, IN_1d_v8,  DA_1d_v8_d, ptr_1d_v8, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                 // if(k == bench_iterations-1)
                 // {
                 //     print2DVectorF(O_v7);
                 //     cout << "-----\n" << endl;
                 // }

                 double elapsed  = 1000*((double)(clock()-t4))/CLOCKS_PER_SEC; // time in milliseconds

                 if(k > 0)
                 t_cpoV8 += elapsed/(Ih*Iw*1.0); // normalized timing
                } // end k loop
      
              // Space
              s_cpo = IN_1d_v8.size() + DA_1d_v8_d.size() + ptr_1d_v8.size();

            } // end if
            else
            {

                // copy data to double
                std::vector<double> DA_1d_v8_d(DA_1d.size(), 0);

                for(int h = 0; h < DA_1d.size(); ++h)
                {
                  DA_1d_v8_d[h] = DA_1d[h];
                }

                for(int k=0;k<bench_iterations;k++)
                {
                    clock_t t5;
                    t5 = clock();
                        // conv_CPO_v7(O_v7, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                      conv_CPO_v7(O_v8, Kernel, IN_1d_v8, DA_1d_v8_d, ptr_1d_v8, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

                      double elapsed  = 1000*((double)(clock()-t5))/CLOCKS_PER_SEC; // time in milliseconds

                      if(k > 0)
                        t_cpoV8 += elapsed/(Ih*Iw*1.0); // normalized timing
                      // cout << "a2: " << t_cpoV8 << " " << elapsed << endl;
                    
                } // end k lop 
                    
                // Space
                s_cpo = IN_1d_v8.size() + DA_1d_v8_d.size() + ptr_1d_v8.size();

            } // end else

              // Include the creation time:
              t_cpoV8 += t_cpo_creation_V8;
        }

#if case_1_time
        cout << "\t\t\tCPO v8" << endl;
        cout << t_cpoV8 << endl;
#endif

        // // New place for CSR:
        // CSR:
          std::vector<int> DA_1d_csr;            
          std::vector<int> IN_1d_csr;
          std::vector<int> ptr_1d_csr(Ow+1, 0);
        {
          // copy data to double

          // CSR Encoding:
          for(int k = 0; k < bench_iterations; ++k)
          {

            clock_t t_cpo_creation_c;
            t_cpo_creation_c = clock();
            CSR(DA_1d_csr, IN_1d_csr, ptr_1d_csr, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
            // CSR2(DA_1d_v7_d, IN_1d_v7_d, ptr_1d_v7, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
            double elapsed  = 1000*((double)(clock()-t_cpo_creation_c))/CLOCKS_PER_SEC; // time in milliseconds
            if(k > 0)
              t_cscc_creation += elapsed/(Ih*Iw*1.0); // normalized timing

            if(k != bench_iterations-1)
            {
              IN_1d_csr.clear();
              DA_1d_csr.clear();
              reset_ptr(ptr_1d_csr,0);
              // cout << " CLEAR CSCC !!!" << endl;
               
            } // end if

            // if (k == bench_iterations - 1)
            // {
                // cout << "CSCC" << endl;
                // cout << "Ptr: " << endl;
                // printVector(ptr_1d_csr);

                // cout << "Data:  " << endl;
                // printVector(DA_1d_csr);
            // }

          } // end for

#if case_1_time
        cout << "\t\t\tCSCC Encoding" << endl;
        cout << t_cscc_creation << endl;
#endif
        }
          // Space:
        s_csr = ptr_1d_csr.size() + DA_1d_csr.size() + IN_1d_csr.size();
        // cout << "ptr : " << ptr_1d_csr.size() << " DA : " << DA_1d_csr.size() << " IN : " << IN_1d_csr.size() << endl;

        // cout << "Index: " << endl;
        // printVector(IN_1d_csr);
        
        // cout << "\n" << endl;
        // cout << t_cpo_creation << endl;
        // cout << t_cscc_creation << endl;
                // Prepare the output for CPO
        vector<vector<float> > O_CSR( Oh , vector<float> (Ow, 0));
        {
            std::vector<double> DA_1d_csr_d(DA_1d_csr.size(), 0);

            for(int h = 0; h < DA_1d_csr.size(); ++h)
            {
              DA_1d_csr_d[h] = DA_1d_csr[h];
            }

            // std::vector<double> DA_1d_v8_d(DA_1d.size(), 0);
            // for(int h = 0; h < DA_1d_v8.size(); ++h)
            // {
            //   DA_1d_v8_d[h] = DA_1d_v8[h];
            // }
        
        
          for(int k = 0;k<bench_iterations;k++)
          {

            clock_t t5;
            t5 = clock();
            
            // 1
          // csrMult_v4(O_CSR, Kernel, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
          // 2
            conv_CPO_v7(O_CSR, Kernel, IN_1d_csr, DA_1d_csr_d, ptr_1d_csr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
            // conv_CPO_v7(O_CSR, Kernel, IN_1d_v8, DA_1d_v8_d, ptr_1d_v8, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           // if(k == bench_iterations-1)
           // {
           //     cout << "CSR: " << endl;
           //     print2DVectorF(O_CSR);
           //     cout << "-----\n" << endl;
           // }
          double elapsed  = 1000*((double)(clock()-t5))/CLOCKS_PER_SEC; // time in milliseconds

          if(k > 0)
            t_csr += elapsed/(Ih*Iw*1.0); // normalized timing
          // cout << "a2: " << t_cpoV8 << " " << elapsed << endl;
            } // end k loop 

          t_csr += t_cscc_creation;

#if case_1_time
    cout << "\t\t\tCSCC Conv" << endl;
    cout << t_csr << endl;
#endif
          // Clear the vectors:
          // DA_1d_csr_d.clear();
          // ptr_1d_csr.clear();
          // IN_1d_csr.clear();

           // Include CPO creation time in this case:
           
        }



        bool s = (t_cpoV8 <= t_csr);
        // elapsed time per feature element in the entire bench iterations
        //        std::cout<<"batch\t"<<In<<"\tdensity\t"<<density<<"\tdensity\t"<<density_cal<<"\tim2col\t"<< t_im2col <<"\tcsr\t"<< t_csr <<"\tcpo\t"<< t_cpo <<std::endl;
        // std::cout << "B-" << bench_iterations << "\t" << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batch\t"<<1
        // <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
        // <<"\tim2col\t"<<t_im2col<<"\tcsr\t" <<t_csr <<"\tcpoV5\t"<< t_cpoV5 <<"\tcpoV6\t"<< t_cpoV6 <<"\tcpoV7\t"<< t_cpoV7 <<"\tcpoV8\t"<< t_cpoV8
        // <<"\tpercentCSSC\t"<< 100.0*(t_im2col-t_csr)/t_im2col  <<"\tpercentV5\t"<< 100.0*(t_im2col-t_cpoV5)/t_im2col <<"\tpercentV6\t"<< 100.0*(t_im2col-t_cpoV6)/t_im2col  
        // <<"\tpercentV7\t"<< 100.0*(t_im2col-t_cpoV7)/t_im2col  <<"\tpercentV8\t"<< 100.0*(t_im2col-t_cpoV8)/t_im2col << "\t" << s << "\n";

        std::cout << "B-" << bench_iterations << "\t" << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batch\t"<<1
        <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
        <<"\tim2col\t"<<t_im2col<<"\tcsr\t" <<t_csr  <<"\tcpoV8\t"<< t_cpoV8
        <<"\tpercentCSSC\t" << 100.0*(t_im2col-t_csr)/t_im2col  <<"\tpercentV8\t"<< 100.0*(t_im2col-t_cpoV8)/t_im2col << "\t" << s << "\n";

        // cout << density_cal <<  "\tt_csr: " << t_csr << "\tt_cpo: " << t_cpoV8 << endl;
        
        ofstream myfile;
        myfile.open ("csr_log.txt", ios::out | ios::app);

        ofstream myfile_encoding;
        myfile_encoding.open ("encoding_log.txt", ios::out | ios::app);
        int batch = 1;

        // // myfile << "B-" << bench_iterations << "\t" << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batch\t"<<1
        // // <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
        // // <<"\tim2col\t"<<t_im2col<<"\tcsr\t" <<t_csr <<"\tcpoV5\t"<< t_cpoV5 <<"\tcpoV7\t"<< t_cpoV7 <<"\tcpoV8\t"<< t_cpoV8
        // // <<"\tpercentCSSC\t"<< 100.0*(t_im2col-t_csr)/t_im2col  <<"\tpercentV5\t"<< 100.0*(t_im2col-t_cpoV5)/t_im2col <<"\tpercentV6\t"<< 100.0*(t_im2col-t_cpoV6)/t_im2col  
        // // <<"\tpercentV7\t"<< 100.0*(t_im2col-t_cpoV7)/t_im2col  <<"\tpercentV8\t"<< 100.0*(t_im2col-t_cpoV8)/t_im2col << "\t" << s << "\n";


        cout << "im2col: " << t_im2col_creation << "\tt_cscc: " << t_cscc_creation << "\tt_cpo_creation: " << t_cpo_creation 
        //<< "\tt_cpo_creation_V6: " << t_cpo_creation_woRep 
        << "\tt_cpo_creationV7: " << t_cpo_creation_V7 << "\tt_cpo_creationV8: " << t_cpo_creation_V8 << endl << endl;
        // exit(0);
        // // Note: t_cpoV6 is t_cpoV3
        myfile << std::setprecision(3) << density  << "\t" << density_cal << "\t" << Kh << "\t" << Kw << "\t" << Ih << "\t" << Iw 
        << "\t" << t_im2col 
        << "\t" << t_csr 
        << "\t" << t_cpoV6
        << "\t" << t_cpoV7 
        << "\t" << t_cpoV8 
        << "\t" << 100.0*(t_im2col-t_csr)/t_im2col 
         << "\t" << 100.0*(t_im2col-t_cpoV6)/t_im2col
        << "\t" << 100.0*(t_im2col-t_cpoV7)/t_im2col 
        << "\t" << 100.0*(t_im2col-t_cpoV8)/t_im2col 
        <<  "\t" << 1.0*s_im2col/s_csr <<"x\t" << 1.0*s_im2col/s_cpo  << "x\n";

        myfile_encoding << std::setprecision(3) << density  << "\t" << density_cal << "\t" << Kh << "\t" << Kw << "\t" << Ih << "\t" << Iw 
        << "\t" << t_im2col_creation 
        << "\t" << t_cscc_creation 
        << "\t" << t_cpo_creation 
        //<< "\t" << t_cpo_creation_woRep 
        << "\t" << t_cpo_creation_V7 
        << "\t" << t_cpo_creation_V8
        << "\t" << 100.0*(t_im2col_creation-t_cscc_creation)/t_im2col_creation 
        << "\t" << 100.0*(t_im2col_creation-t_cpo_creation)/t_im2col_creation 
        // << "\t" << 100.0*(t_im2col_creation-t_cpo_creation_woRep)/t_im2col_creation 
        << "\t" << 100.0*(t_im2col_creation-t_cpo_creation_V7)/t_im2col_creation 
        << "\t"<< 100.0*(t_im2col_creation-t_cpo_creation_V8)/t_im2col_creation  << "\n";


        // This is for checking
        //  myfile << std::setprecision(3) << density  << "\t" << density_cal << "\t" << Kh << "\t" << Kw << "\t" << Ih << "\t" << Iw << 
        // "\t" << t_im2col << "\t" << t_csr << "\t" << t_cpoV6 << "\t" << t_cpoV8 << 
        // "\t"<< 100.0*(t_im2col-t_csr)/t_im2col <<"\t" << 100.0*(t_im2col-t_cpoV6)/t_im2col << "\t" << 100.0*(t_im2col-t_cpoV8)/t_im2col 
        // <<  "\t" << 1.0*s_im2col/s_csr <<"x\t" << 1.0*s_im2col/s_cpo  << "x\n";
        // myfile.close();

        

        
    } // density loop

    ofstream myfile;
    myfile.open ("csr_log.txt", ios::out | ios::app);
    myfile << "\n";
    myfile.close();

    ofstream myfile_encoding;
    myfile_encoding.open ("encoding_log.txt", ios::out | ios::app);
    myfile_encoding << "\n";
    myfile_encoding.close();
  } // end I loop

    ofstream myfile;
    myfile.open ("csr_log.txt", ios::out | ios::app);
    myfile << "\n";
    myfile.close();

    ofstream myfile_encoding;
    myfile_encoding.open ("encoding_log.txt", ios::out | ios::app);
    myfile_encoding << "\n";
    myfile_encoding.close();
} // end K loop
    
    return 0;
}
