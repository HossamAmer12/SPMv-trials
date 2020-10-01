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

void printVector(std::vector<int>& x);
void CPO(MatrixXf& lowered_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> > &IN,
         vector<vector<int> > &DA, vector<vector<int> > &ptr);

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


void transform2dTo1d(vector<vector<int> >  &IN,  vector<vector<int> > &DA, vector<vector<int> >  &ptr, vector<int>  &IN_1d,  vector<int> &DA_1d, vector<int>  &ptr_1d)
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

void CPO_Encoding(std::vector<int> &IN_1d, std::vector<int> &DA_1d, std::vector<int> &ptr_1d, MatrixXf& org_fm, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
  int n = ceil(Kw / Sw);
  if (Kw % Sw == 0)
  {
      n = Kw / Sw;
  }
  
  vector<vector<int> > IN(n); // n is the rows
  vector<vector<int> > DA(n); // n is the rows
  vector<vector<int> > ptr( n , vector<int> (Ow + 1, 0)); // n is the rows
  CPO(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr);

  // transform 2d to 1d:
  int count_ptr = n*(1+Ow);

  // Get the total number of non zeros: we can save it while encoding
  int count_d = 0;
  for(int i = 0 ; i < DA.size(); ++i)
  {
      count_d += DA[i].size();
  } 

  IN_1d.reserve(count_d);
  DA_1d.reserve(count_d);
  ptr_1d.reserve(count_ptr);  

  transform2dTo1d(IN, DA, ptr, IN_1d, DA_1d, ptr_1d);
}


void CPO_EncodingV7(std::vector<int> &IN_1d, std::vector<int> &DA_1d, std::vector<int> &ptr_1d, MatrixXf& org_fm, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
  int n = ceil(Kw / Sw);
  if (Kw % Sw == 0)
  {
      n = Kw / Sw;
  }
  
  vector<vector<int> > IN(n); // n is the rows
  vector<vector<int> > DA(n); // n is the rows
  vector<vector<int> > ptr( n , vector<int> (Ow + 1, 0)); // n is the rows
  CPO(org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw, IN, DA, ptr);
       
  // transform 2d to 1d:
  int count_ptr = n*(1+Ow);
  if(n != 1)
  {
    count_ptr = (n-1)*(1 + Ow) + min(int(ptr[0].size()), 3);
  }


  // Get the total number of non zeros: we can save it while encoding
  int count_d = 0;
  for(int i = 0 ; i < DA.size(); ++i)
  {
      count_d += DA[i].size();
  } 

  IN_1d.reserve(count_d);
  DA_1d.reserve(count_d);
  ptr_1d.reserve(count_ptr);  

  if(n != 1)
  {
    transform2dTo1dV7(IN, DA, ptr, IN_1d, DA_1d, ptr_1d);  
  }
  else
  {
    transform2dTo1d(IN, DA, ptr, IN_1d, DA_1d, ptr_1d);
  }

  // cout << "After: " << endl;
  // printVector(ptr_1d);
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
void CSR(std::vector<double> &DA, std::vector<int> &IN, std::vector<int> &ptr, MatrixXf& org_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
{
    int i = 0;
    int x = 0;
    int m = 0;

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
                if (j == Iw - 3)
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
    // int bench_iterations = 1000;
    // int bench_iterations = 100;
  // int bench_iterations = 1;

  // std::vector<int> I_list = {8,17,50};
  // std::vector<int> Kh_list = {3, 7, 1};
  // std::vector<int> Kw_list = {3,1, 7};


  // Big test
  // std::vector<int> I_list = {8,17,50};
  // std::vector<int> Kh_list = {3, 1, 3, 7, 1};
  // std::vector<int> Kw_list = {3, 3, 1, 1, 7};

  std::vector<int> I_list = {8};
  std::vector<int> Kh_list = {3};
  std::vector<int> Kw_list = {3};

  // std::vector<int> I_list = {17};
  // std::vector<int> Kh_list = {3};
  // std::vector<int> Kw_list = {3};


  // std::vector<int> I_list = {17};
  // std::vector<int> Kh_list = {7};
  // std::vector<int> Kw_list = {1};

  // std::vector<int> I_list = {8};
  // std::vector<int> Kh_list = {3};
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
        // float density = 1;
    //    float density = 0.3;
  // float density = 0.1;
    

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
        float t_cpo_creation_V7 = 0;

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
        
        
        
        int start_row_int = 0;
        int start_col_int = 0;
        
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
        std::vector<int> ptr_1d;

        for(int k = 0; k < bench_iterations; ++k)
        {

          clock_t t_cpo_creation_c;
          t_cpo_creation_c = clock();
          CPO_Encoding(IN_1d, DA_1d, ptr_1d, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
          double elapsed  = 1000*((double)(clock()-t_cpo_creation_c))/CLOCKS_PER_SEC; // time in milliseconds
          if(k > 0)
            t_cpo_creation += elapsed/(Ih*Iw*1.0); // normalized timing

          if(k != bench_iterations-1)
          {
            IN_1d.clear();
            DA_1d.clear();
            ptr_1d.clear();  
          }
          
        }

        // cout << t_cpo_creation << endl;
        // cout << "Ptr: " << endl;
        // printVector(ptr_1d);

        // cout << "Data:  " << endl;
        // printVector(DA_1d);

        // cout << "Index: " << endl;
        // printVector(IN_1d);

        // cout << "\nPtr2: ";
        // print2DVector(ptr);
    
        // cout << "\n\nIN: ";
        // print2DVector(IN);
    
        // cout << "\n\nData:";
        // print2DVector(DA);
        // cout << "\n" << endl;
        // cout << t_cpo_creation << endl;

        ///////////

        // V3 Code:
         // Perform 50 times raw sparse matrix dense vector multiplication: d_CPO = d_m * d_b
        {
            // Prepare the output for CPO
            vector<vector<float> > O_CPO2( Oh , vector<float> (Ow, 0));
            
            for(int k=0;k<bench_iterations;k++){
                


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

        
        // V7 Code:
          // Remove repeats and transofrm 2d to 1d:
      //   int count_ptr_v7 = 0;
      //   for(int i = 0 ; i < ptr.size(); ++i)
      //   {

      //     // if it one ptr, don't do anything
      //     if(i == 0 && n != 1)
      //     {
      //       int f = ptr[i].size();
      //       count_ptr_v7 += min(f, 3);   
      //     }
      //     else
      //     {
      //       count_ptr_v7 += ptr[i].size();
      //     }
          
      //   } 

      //   int count_d_v7 = 0;
      //   for(int i = 0 ; i < DA.size(); ++i)
      //   {

      //     count_d_v7 += DA[i].size();
      //   }   

      // std::vector<int> IN_1d_v7(count_d_v7, 0);
      // std::vector<int> DA_1d_v7(count_d_v7, 0);
      // std::vector<int> ptr_1d_v7(count_ptr_v7, 0);

      // transform2dTo1dv1(IN, DA, ptr, IN_1d_v7, DA_1d_v7, ptr_1d_v7);

        // CPO Encoding V7:
        std::vector<int> IN_1d_v7;
        std::vector<int> DA_1d_v7;
        std::vector<int> ptr_1d_v7;

        for(int k = 0; k < bench_iterations; ++k)
        {

          clock_t t_cpo_creation_c;
          t_cpo_creation_c = clock();
          CPO_EncodingV7(IN_1d_v7, DA_1d_v7, ptr_1d_v7, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
          double elapsed  = 1000*((double)(clock()-t_cpo_creation_c))/CLOCKS_PER_SEC; // time in milliseconds
          if(k > 0)
            t_cpo_creation_V7 += elapsed/(Ih*Iw*1.0); // normalized timing

          if(k != bench_iterations-1)
          {
            IN_1d_v7.clear();
            DA_1d_v7.clear();
            ptr_1d_v7.clear();  
          }
          
        }

        // cout << t_cpo_creation_V7 << endl;
        // cout << "Ptr: " << endl;
        // printVector(ptr_1d);

        // cout << "New Ptr: " << endl;
        // printVector(ptr_1d_v7);

        // cout << "Index: " << endl;
        // printVector(IN_1d);
        // cout << "New Index: " << endl;
        // printVector(IN_1d_v7);
        // exit(0);
           
        // CPO V8:
        {
            // Prepare the output for CPO
            vector<vector<float> > O_v8( Oh , vector<float> (Ow, 0));
            
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
                  conv_CPO_v8_trim(O_v8, Kernel, IN_1d_v7,  DA_1d_v7_d, ptr_1d_v7, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

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
              conv_CPO_v7(O_v8, Kernel, IN_1d,  DA_1d_v7_d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

              double elapsed  = 1000*((double)(clock()-t5))/CLOCKS_PER_SEC; // time in milliseconds

              if(k > 0)
                t_cpoV8 += elapsed/(Ih*Iw*1.0); // normalized timing
              // cout << "a2: " << t_cpoV8 << " " << elapsed << endl;
            
            } // end k lop 
                    
            // Space
            s_cpo = IN_1d.size() + DA_1d_v7_d.size() + ptr_1d.size();

            } // end else

              // Include the creation time:
              t_cpoV8 += t_cpo_creation_V7;
        }



        // // New place for CSR:
        // CSR:
        {
            // Prepare the output for CPO
            vector<vector<float> > O_CSR( Oh , vector<float> (Ow, 0));
          
            if(n != 1)
            {
                
                // copy data to double
              std::vector<double> DA_1d_csr_d;            
              std::vector<int> IN_1d_csr;
              std::vector<int> ptr_1d_csr;

              // CSR Encoding:
              for(int k = 0; k < bench_iterations; ++k)
              {

                clock_t t_cpo_creation_c;
                t_cpo_creation_c = clock();
                CSR(DA_1d_csr_d, IN_1d_csr, ptr_1d_csr, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                // CSR2(DA_1d_v7_d, IN_1d_v7_d, ptr_1d_v7, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                double elapsed  = 1000*((double)(clock()-t_cpo_creation_c))/CLOCKS_PER_SEC; // time in milliseconds
                if(k > 0)
                  t_cscc_creation += elapsed/(Ih*Iw*1.0); // normalized timing

                if(k != bench_iterations-1)
                {
                  IN_1d_csr.clear();
                  DA_1d_csr_d.clear();
                  ptr_1d_csr.clear();  
                } // end if

              } // end for

              // Space:
              s_csr = ptr_1d_csr.size() + DA_1d_csr_d.size() + IN_1d_csr.size();
             
            // cout << "Ptr: " << endl;
            // printVector(ptr_1d_v7);

            // cout << "Data:  " << endl;
            // printVectorD(DA_1d_v7_d);

            // cout << "Index: " << endl;
            // printVector(IN_1d_v7_d);
            
            // cout << "\n" << endl;
            // cout << t_cpo_creation << endl;
            // cout << t_cscc_creation << endl;

              for(int k = 0;k<bench_iterations;k++)
              {

                clock_t t5;
                t5 = clock();
                
                // 1
              // csrMult_v4(O_CSR, Kernel, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
              // 2
                conv_CPO_v7(O_CSR, Kernel, IN_1d_csr, DA_1d_csr_d, ptr_1d_csr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

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

              // Clear the vectors:
              DA_1d_csr_d.clear();
              ptr_1d_csr.clear();
              IN_1d_csr.clear();

               // Include CPO creation time in this case:
               t_csr += t_cscc_creation;

            } // end if

            else
            {

                // copy data to double
                std::vector<double> DA_1d_csr_d(DA_1d.size(), 0);

                for(int h = 0; h < DA_1d.size(); ++h)
                {
                  DA_1d_csr_d[h] = DA_1d[h];
                }

                for(int k=0;k<bench_iterations;k++)
                {
                  clock_t t5;
                  t5 = clock();
                // conv_CPO_v7(O_v7, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                  conv_CPO_v7(O_CSR, Kernel, IN_1d, DA_1d_csr_d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
                  double elapsed  = 1000*((double)(clock()-t5))/CLOCKS_PER_SEC; // time in milliseconds

                  if(k > 0)
                    t_csr += elapsed/(Ih*Iw*1.0); // normalized timing
                  // cout << "a2: " << t_cpoV8 << " " << elapsed << endl;

               } // end k loop 
                    
              // Include CPO creation time in this case:
               t_csr += t_cpo_creation;

            } // end else
        } // end CSR brace


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
        int batch = 1;

        // // myfile << "B-" << bench_iterations << "\t" << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batch\t"<<1
        // // <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
        // // <<"\tim2col\t"<<t_im2col<<"\tcsr\t" <<t_csr <<"\tcpoV5\t"<< t_cpoV5 <<"\tcpoV7\t"<< t_cpoV7 <<"\tcpoV8\t"<< t_cpoV8
        // // <<"\tpercentCSSC\t"<< 100.0*(t_im2col-t_csr)/t_im2col  <<"\tpercentV5\t"<< 100.0*(t_im2col-t_cpoV5)/t_im2col <<"\tpercentV6\t"<< 100.0*(t_im2col-t_cpoV6)/t_im2col  
        // // <<"\tpercentV7\t"<< 100.0*(t_im2col-t_cpoV7)/t_im2col  <<"\tpercentV8\t"<< 100.0*(t_im2col-t_cpoV8)/t_im2col << "\t" << s << "\n";

        // // Note: t_cpoV6 is t_cpoV3
        myfile << std::setprecision(3) << density  << "\t" << density_cal << "\t" << Kh << "\t" << Kw << "\t" << Ih << "\t" << Iw << 
        "\t" << t_im2col << "\t" << t_csr << "\t" << t_cpoV6 << "\t" << t_cpoV8 << 
        "\t"<< 100.0*(t_im2col-t_csr)/t_im2col <<"\t" << 100.0*(t_im2col-t_cpoV6)/t_im2col << "\t" << 100.0*(t_im2col-t_cpoV8)/t_im2col 
        <<  "\t" << 1.0*s_im2col/s_csr <<"x\t" << 1.0*s_im2col/s_cpo  << "x\n";

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
  } // end I loop

    ofstream myfile;
    myfile.open ("csr_log.txt", ios::out | ios::app);
    myfile << "\n";
    myfile.close();
} // end K loop
    
    return 0;
}

