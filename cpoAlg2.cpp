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

void print2DVectorF(std::vector<vector<float>>& x);
void printVector(std::vector<int>& x);


void transform2dTo1dv1(vector<vector<int> >  &IN,  vector<vector<int> > &DA, vector<vector<int> >  &ptr, vector<int>  &IN_1d,  vector<int> &DA_1d, vector<int>  &ptr_1d)
{ 

          cout << "Total size: " << ptr_1d.size() << endl;
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

// New v9 with proper unrilling:
void conv_CPO_v9_trim(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<int> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
 
    const int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    const int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr
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

       // Submat 0
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

       // x loop
      // l loop
            // cout << "V9) Sumbat: " << 0 << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
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


      // cout << "V9) Sumbat: " << (number-1) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
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

    // For each ptr type
    // int type_ptr = 0;
    for (int type_ptr = 1; type_ptr < n-1; ++type_ptr)
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
        int used_data   = *Adata_help;  Adata_help++;

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
              O[y_out][0 + i] += used_data * K[kernel_common_index - i]; 

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
        int used_data   = *Adata_help;  Adata_help++;

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
            int out_common_index     = number - 1 - type_ptr;

            for(int i = 1; i <= type_ptr; ++i)
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
    type_ptr = n-1;
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


} // end conv_CPO_v


// Old v9 without proper unrolling:
void conv_CPO_v9(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<int> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    const int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    const int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;
    
    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    int *x_ptr           = &ptr[0];
    int  x               = *x_ptr;
    int *Aindex_help     = &IN[x];
    int *Adata_help      = &DA[x];

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
        int used_data   = *Adata_help;  Adata_help++;

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
              O[y_out][0 + i] += used_data * K[kernel_common_index - i]; 

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
        int used_data   = *Adata_help;  Adata_help++;

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
              O[y_out][number - 1 + i - type_ptr] += used_data * K[kernel_common_index - i]; 

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


                 // O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 
                 // O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 
                 
                 // O[y_out][x_out + i] += used_data * K[kernel_common_index - i]; 

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


void conv_CPO_v8(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<int> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    const int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    const int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr
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

       // Submat 0
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

       // x loop
      // l loop
            cout << "V8) Sumbat: " << 0 << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
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


      cout << "V8) Sumbat: " << (number-1) << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;    
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

      cout << "V8) Sumbat: " << submat << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;   
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



void conv_CPO_v7(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<int> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;

    const int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    const int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr
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


void conv_CPO_v4(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<int> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
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

    // Ptry_type: 0
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
      
        // for(int i = 0; i <= type_ptr; ++i)
        // {

          // Loop on Kh for the output
        for(int l = 0; l < Kh; ++l)
        {
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int input_index  = used_index;
            int y_out        = (input_index/Kw) - l;
            int x_out        = submat;
        
           if(y_out < 0 || y_out >= Oh){
            // cout << "continue YYY============\n" << endl;
            continue;
         }
        
        // cout << "R) " << y_out << ", C) " << x_out << ", Data: " << DA[type_ptr][x] << ", Index: " << input_index  << ", ac_Index: " << IN[type_ptr][x] << endl;
       //    // O(y_out, x_out) += DA[type_ptr][x] * 1.0;
          O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
      
          } // for each l in Kh
        // } // end i
      } // end x
    } // end sumbat

    ++x_ptr;
  } // end type ptr

    // Ptry_type: 1:
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
      
        // Unroll the loop
        for(int l = 0; l < Kh; ++l)
        {
            cout << " LLLLLLLLLLL: " << l << endl;
              // I = 0:
            int i = 0;
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int input_index  = used_index - i;
            int y_out        = (input_index/Kw) - l;
            int x_out        = i + submat;
            
            bool flag = true;
            cout << i << ") X_out: " << x_out << " Y_out: " << y_out << endl;

            if(y_out >= 0 && y_out < Oh){
              cout << "Hit YYY============" << endl;
              O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
            }
            else
            {
              flag = false;
            }

            if(!flag) continue;

             // I = 1:
            i = 1;
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            input_index  = used_index - i;
            y_out        = (input_index/Kw) - l;
            x_out        = i + submat;
          
            // cout << i << ") X_out: " << x_out << " Y_out: " << y_out << endl;

            if(y_out >= 0 && y_out < Oh){
              cout << "Hit YYY============" << endl;
              O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
              // if(flag == false)
              // {
              //   cout << "YES I AM HERE TO KILL" << endl;
              // }
            }

       } // end l loop

      } // end x
    } // end sumbat

    ++x_ptr;
  } // end type ptr

    // Ptry_type: 2
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
        
        // Unroll the loop
        for(int l = 0; l < Kh; ++l)
        {
            cout << " LLLLLLLLLLL: " << l << endl;
              // I = 0:
            int i = 0;
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            int input_index  = used_index - i;
            int y_out        = (input_index/Kw) - l;
            int x_out        = i + submat;
            
            bool flag = true;
            cout << i << ") X_out: " << x_out << " Y_out: " << y_out << endl;

            if(y_out >= 0 && y_out < Oh) {
               cout << "Hit YYY============" << endl;
               O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];

               // int input_index  = used_index - 1;
               // O[y_out][x_out+1] += used_data * K[input_index%Kw + l*Kw];

               // input_index  = used_index - 2;
               // O[y_out][x_out+2] += used_data * K[input_index%Kw + l*Kw];

               for(int i = 1; i <= type_ptr; ++i)
               {
                 int input_index     = used_index - i;
                 O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 

               }
          }
            /*
            if(!flag) continue;

             // I = 1:
            i = 1;
             // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            input_index  = used_index - i;
            y_out        = (input_index/Kw) - l;
            x_out        = i + submat;
          
            // cout << i << ") X_out: " << x_out << " Y_out: " << y_out << endl;

            if(y_out >= 0 && y_out < Oh){
              cout << "Hit YYY============" << endl;
              O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];

              // if(flag == false)
              // {
              //   cout << "YES I AM HERE TO KILL" << endl;
              // }
            }
            else
            {
              // cout << " No more here" << endl;
              continue;
            }

            // I: 1
            i = 2;
            // cout << "Use x:  " << i << ", with ind: " << used_index << endl;
            input_index  = used_index - i;
            y_out        = (input_index/Kw) - l;
            x_out        = i + submat;
          
            cout << i << ") X_out: " << x_out << " Y_out: " << y_out << endl;
         //   if(y_out < 0 || y_out >= Oh){
         //    cout << "continue YYY============\n" << endl;
         //    continue;
         // }
         // else
         // {
         //    if(flag == false)
         //      {
         //        cout << "YES I AM HERE TO KILL" << endl;
         //      }
         // }

         O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
        */


       } // end l loop

      } // end x
    } // end sumbat

    ++x_ptr;
  } // end type ptr

    // cout << "Output: " << endl;
    // print2DVectorF(O);
    // cout << "-----\n" << endl;
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

void conv_CPO_v2(vector<vector<float> > & O, vector<int> const &K, vector<vector<int> >  &IN,  vector<vector<int> > &DA, vector<vector<int> >  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;
    



    int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;
    
    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";


    // For each ptr type
    // int type_ptr = 0;
    for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    {

      int x                = ptr[type_ptr][0];
      int *Aindex_help     = &IN[type_ptr][x];
      int *Adata_help      = &DA[type_ptr][x];

    // For each submat
    for (int submat = 0; submat < number; ++submat)
    {
     
      cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << endl;
      for(; x < ptr[type_ptr][submat+1]; ++x)
      {      
       cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  

        // How many time to iterate?
        int used_index  = *Aindex_help; Aindex_help++;
        int used_data   = *Adata_help; Adata_help++;

        // int shereet2 = min(submat, type_ptr); 
        for(int i = 0; i <= type_ptr; ++i)
        {

          cout << "Use x:  " << i << ", with ind: " << used_index << endl;

          // Loop on Kh for the output
          for(int l = 0; l < Kh; ++l)
          {
            int input_index  = used_index - i;
            int y_out        = (input_index)/Kw - l;
            int x_out        = i + submat;
        
          if(y_out < 0 || y_out >= Oh){
            // cout << "continue YYY============\n" << endl;
            continue;
         }
        
    
        cout << "R) " << y_out << ", C) " << x_out << ", Data: " << DA[type_ptr][x] << ", Index: " << input_index  << ", ac_Index: " << IN[type_ptr][x] << endl;
       //    // O(y_out, x_out) += DA[type_ptr][x] * 1.0;
          O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
      
       } // for each l in Kh
        } // end i
      } // end x
    } // end sumbat
  } // end type ptr

    // cout << "Output: " << endl;
    // print2DVectorF(O);
    // cout << "-----\n" << endl;
}



void conv_CPO_v1(vector<vector<float> > & O, vector<int> const &K, vector<vector<int> > const &IN,  vector<vector<int> > const &DA, vector<vector<int> > const &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;
    
    int n      = ceil(Kw/Sw); // n is the number of ptr (NPO, PO2, PO3)
    int number = floor((Iw - Kw)/Sw) + 1; // number of elements in each ptr
    // number = 0; n = 1;
    
    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";


    // For each ptr type
    // int type_ptr = 0;
    for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    // for (int type_ptr = 0; type_ptr < 1; ++type_ptr)
    // for (int type_ptr = 2; type_ptr < 3; ++type_ptr)
    {

    // For each submat
    for (int submat = 0; submat < number; ++submat)
    {
     
      cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << endl;
      for(int x = ptr[type_ptr][submat]; x < ptr[type_ptr][submat+1]; ++x)
      {      
       cout << "\n" << type_ptr << " ==> Current Submat " << submat << ", ptr:  " <<  x <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;  

        // How many time to iterate?
        int used_index  = IN[type_ptr][x];
        int used_data   = DA[type_ptr][x];
        int shereet     = (type_ptr > 0)? 1:0;

        // int shereet2 = min(submat, type_ptr); 
        for(int i = 0; i <= type_ptr; ++i)
        {

          cout << "Use x:  " << i << ", with ind: " << used_index << endl;

          // Loop on Kh for the output
          for(int l = 0; l < Kh; ++l)
          {
            int input_index  = used_index - i;
            int y_out        = (input_index)/Kw - l;
            int x_out        = i + submat;
        
          if(y_out < 0 || y_out >= Oh){
            // cout << "continue YYY============\n" << endl;
            continue;
         }
        
    
        cout << "R) " << y_out << ", C) " << x_out << ", Data: " << DA[type_ptr][x] << ", Index: " << input_index  << ", ac_Index: " << IN[type_ptr][x] << endl;
       //    // O(y_out, x_out) += DA[type_ptr][x] * 1.0;
          O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];
      
       } // for each l in Kh
        } // end i
      } // end x
    } // end sumbat
  } // end type ptr

    // cout << "Output: " << endl;
    // print2DVectorF(O);
    // cout << "-----\n" << endl;
}


void conv(vector<vector<float> > & O, vector<int> const &K, vector<vector<int> > const &IN,  vector<vector<int> > const &DA, vector<vector<int> > const &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
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
    
      // int shereet2 = (submat == 0)? 0:type_ptr;
      // shereet2     = (submat == 1)? 1:shereet2;   

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
      for(int i = 0; i <= shereet2; ++i)
      {
  
       // From ptr r t r+1
        int shereet = (type_ptr > 0)? 1:0;
  //     int shereet = (type_ptr > 0 and submat > 0)? 1:0;
       // int shereet = 1;

       for(int x = ptr[type_ptr][submat-i*shereet]; x < ptr[type_ptr][submat-i*shereet+1]; ++x)
       {
        
         
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
    
      // int shereet2 = (submat == 0)? 0:type_ptr;
      // shereet2     = (submat == 1)? 1:shereet2; 
        // How many time to iterate?
      int shereet2;
      if(type_ptr == 0)
      {
        shereet2 = 0;
      }
      else
      {
        if(submat == 0)
         {
          shereet2 = 0;
         } 
         else
         {
          if(submat == 1)
          {
            shereet2 =  1;
          }
          else
          {
            shereet2 = type_ptr;
          }
           
         }
      }
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
            cout << "continue YYY============\n" << endl;
            continue;
         }
      
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



void conv_v2(vector<vector<float> > & O, vector<int> const &K, vector<vector<int> > const &IN,  vector<vector<int> > const &DA, vector<vector<int> > const &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
{
    // cout << "Shape " << O.rows() << ", " << O.cols() << endl;
    
    int n      = ceil(Kw/Sw);
    int number = floor((Iw - Kw)/Sw) + 1;
    
    // cout << "# of submatrices: " << number << ", # of ptrs: " << n << "\n\n\n";

    // For each ptr type
    for (int type_ptr = 0; type_ptr < n; ++type_ptr)
    {
	// For each submat
	for (int submat = 0; submat < number; ++submat)
	{
	  
	  for(int i = 0; i <= type_ptr; ++i)
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

    // cout << "\n===CPO Output with Size: " << O.rows() << ", " << O.cols() <<  " \n" << O << std::endl;
    print2DVectorF(O);
    cout << "-----\n" << endl;

}



void bench_Sparse(const SparseMatrix<float> &m, const MatrixXf &in, MatrixXf &o) {
    // o.noalias() = m*in.transpose();
    o.noalias() = m*in;
}

void bench_Dense(const MatrixXf &m, const MatrixXf &in, MatrixXf &o) {
    // o.noalias() = m*in.transpose();
    o.noalias() = m*in;
    
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

void reset2DVector(std::vector<vector<float>>& x)
{
    for(int i = 0; i < x.size(); ++i)
    {
        for(int j = 0; j < x[i].size(); ++j)
        {
            x[i][j] = 0;
        }
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


void CPO(MatrixXf& O, VectorXf& K, MatrixXf& lowered_mat, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw, vector<vector<int> > &IN,  vector<vector<int> > &DA, vector<vector<int> > &ptr)
{

    std::cout << "\n===Lowered Feature Map of Size: " << lowered_mat.rows() << ", " << lowered_mat.cols() << "\n" << lowered_mat << std::endl;

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

        //if (l < 0)
       // {
         //   break;
        //}
    } // end for (j = Iw - Kw; Iw; ++j)

    cout << "\nPtr: ";
    print2DVector(ptr);

    cout << "\n\nIN: ";
    print2DVector(IN);

    cout << "\n\nData:";
    print2DVector(DA);
    cout << "\n" << endl;

}

int main()
{
    int FREQ = 1; 

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
     

    Ih = Iw = 8;
    // int Ic = 32; // this is for the node
    int Ic = 1; // put it as articial for now
    int In = 1;
    
    int K = 1; // number of filters
    
    // Kernel dimensions
    int Kh = 3;
    int Kw = 3;
    
    int Oh = (1 + Ih - Kh + 2 * padding)/stride; // removed + 1
    int Ow = (1 + Iw - Kw + 2 * padding)/stride;
    
    cout << "Ouptut dimensions: " << Oh << ", " << Ow << endl;

    int iter = 1;  // total number of times to perform the test for each of dense, sparse multiplication
    
    // Create your original input feature map:
    MatrixXf org_fm = MatrixXf::Zero(Ih, Iw);
    
    std::vector<int> cols = {0,1,4,0,4,0,4};
    std::vector<int> rows = {0,0,0,2,2,3,3};
    std::vector<double> values = {1,1,1,1,1,1,1};
    
    
    org_fm(2, 0) = 1;
    org_fm(4, 1) = 3;
    org_fm(1, 2) = 2;
    org_fm(2, 2) = 1;
    org_fm(0, 3) = 2;
    org_fm(4, 3) = 2;
    org_fm(1, 4) = 3;
    org_fm(3, 4) = 1; 
    /**/
    /*
    0 0 0 0 1 1 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 1
    0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0
*/
    /*
    org_fm(7, 1) = 1;
    org_fm(4, 2) = 1;
    org_fm(0, 5) = 1;
    org_fm(0, 6) = 1;
    org_fm(3, 6) = 1;
    org_fm(3, 7) = 1;
    */
    
    // org_fm(2, 0) = 1;
    // org_fm(7, 1) = 1;
    // org_fm(2, 2) = 1;
    // org_fm(0, 7) = 1;
    /*
    0 0 0 0 0 0 1 0
    0 0 1 0 0 0 0 0
    1 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0
    */
    /*
    for(int i=0; i < cols.size(); i++)
    {
        // org_fm(rows[i], cols[i])        = values[i];
        org_fm(rows[i], cols[i])        = 1;
    }
    */
    
    // Print out the original feature map:
    std::cout << "\n===Original Feature Map: \n" << org_fm << std::endl;
    cout << "-----\n" << endl;
    
    
    // Create the filter K and its vectorized version:
    MatrixXf filter             = MatrixXf::Ones(Kh, Kw);
    VectorXf filter_vectorized  = VectorXf::Ones(Kh*Kw);
    // filter_vectorized(0) = 12;
    
    // Print out the im2col interedmiate feature map:
    std::cout << "\n===Filter: " <<  " \n" << filter_vectorized  << std::endl;
    cout << "-----\n" << endl;
    
    // Prepare the output for im2col, sparseMat, CPO
    MatrixXf d_o1 = MatrixXf::Zero(Oh, Ow);
    MatrixXf d_o2 = MatrixXf::Zero(Oh, Ow);
    MatrixXf d_o3 = MatrixXf::Zero(Oh, Ow);
   	 
    int n;
    if (Kw % Sw == 0)
    {
        n = Kw / Sw;
    }
    else
    {
        n = ceil(Kw / Sw);
    }

    vector<vector<int> > IN(n); // n is the rows
    vector<vector<int> > DA(n); // n is the rows
    vector<vector<int> > ptr(n); // n is the rows
	
   
   
    // Create the Kernel
    vector<int> Kernel(Kh*Kw, 1);	

    


    // Perform 50 times raw sparse matrix dense vector multiplication of CPO: d_o3 = d_m * d_b
    {
        // clock_t t;
        // t = clock();
        for(int k=0;k<1;k++) CPO(d_o3, filter_vectorized, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw,
				IN, DA, ptr);




      // transform 2d to 1d:
      // int count_ptr = 0;
      // for(int i = 0 ; i < ptr.size(); ++i)
      //   {

      //     count_ptr += ptr[i].size();
         
      //   } 

      //    int count_d = 0;
      //   for(int i = 0 ; i < DA.size(); ++i)
      //   {

      //     count_d += DA[i].size();
      //   } 

    //     // 2d to 1d With repeats CPO
    //     std::vector<int> IN_1d(count_d, 0);
    //     std::vector<int> DA_1d(count_d, 0);
    //     std::vector<int> ptr_1d(count_ptr, 0);

    //      transform2dTo1d(IN, DA, ptr, IN_1d, DA_1d, ptr_1d);
    //      ///////////
        
	   //  for(int k=0;k<FREQ;k++)
		  // {            

    //         // Create the output
    //         vector<vector<float> > O( Oh , vector<float> (Ow, 0));  

    //         // cout << "\nPtr: ";
    //         // print2DVector(ptr);

    //         // cout << "\n\nIN: ";
    //         // print2DVector(IN);

    //         // cout << "\n\nData:";
    //         // print2DVector(DA);
    //         // cout << "\n" << endl;

    //         clock_t t;
    //         t = clock();
		  //       // conv(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
    //         // conv_CPO_v1(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

    //         // conv_CPO_v3(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

    //         // V4:
    //         conv_CPO_v4(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
    //         double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
    //         t_csr+=elapsed/(Ih*Iw*1.0); // normalized timing


    //         if(k == FREQ - 1)
    //         {
    //             cout << "Output: " << endl;
    //             print2DVectorF(O);    
    //         }
    //     }
    // }


        // 2d to 1d without repeats CPO
         int count_ptr = 0;
        for(int i = 0 ; i < ptr.size(); ++i)
        {

          if(i == 0)
          {
            int f = ptr[i].size();
            count_ptr += min(f, 3);  
          }
          else
          {
            count_ptr += ptr[i].size();
          }
          
        } 

         int count_d = 0;
        for(int i = 0 ; i < DA.size(); ++i)
        {

          count_d += DA[i].size();
        }   

        std::vector<int> IN_1d(count_d, 0);
        std::vector<int> DA_1d(count_d, 0);
        std::vector<int> ptr_1d(count_ptr, 0);

         transform2dTo1dv1(IN, DA, ptr, IN_1d, DA_1d, ptr_1d);
         ///////////
        
      for(int k=0;k<FREQ;k++)
      {            

            // Create the output
            vector<vector<float> > O( Oh , vector<float> (Ow, 0));  

            // cout << "\nPtr: ";
            // print2DVector(ptr);

            // cout << "\n\nIN: ";
            // print2DVector(IN);

            // cout << "\n\nData:";
            // print2DVector(DA);
            // cout << "\n" << endl;

            clock_t t;
            t = clock();
            // conv(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
            // conv_CPO_v1(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

            // conv_CPO_v3(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

            // V4:
            // conv_CPO_v7(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
            conv_CPO_v8(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
            double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
            t_csr+=elapsed/(Ih*Iw*1.0); // normalized timing


            if(k == FREQ - 1)
            {
                cout << "Output: " << endl;
                print2DVectorF(O);    
            }
        }
    }


    int count_d = 0;
    for(int i = 0 ; i < DA.size(); ++i)
    {

      count_d += DA[i].size();
    }   

    int count_ptr_v9 = 0;
    for(int i = 0 ; i < ptr.size(); ++i)
    {

      if(i < ptr.size() - 1)
      {
        int f = ptr[i].size();
        count_ptr_v9 += min(f, 3);  
      }
      else
      {
        count_ptr_v9 += ptr[i].size();
      }
      
    }


    // Conv CPO v9:
    {

    std::vector<int> IN_1d_v9(count_d, 0);
    std::vector<int> DA_1d_v9(count_d, 0);
    std::vector<int> ptr_1d_v9(count_ptr_v9, 0);


    // cout << "Ptr: ";
    // print2DVector(ptr);
    transform2dTo1dv9(IN, DA, ptr, IN_1d_v9, DA_1d_v9, ptr_1d_v9);
    
    ///////////
    // cout << "Ptr: ";
    // printVector(ptr_1d_v9);

    for(int k=0;k<FREQ;k++)
    {            

            // Create the output
            vector<vector<float> > O_v9( Oh , vector<float> (Ow, 0));  

            // cout << "\nPtr: ";
            // print2DVector(ptr);

            // cout << "\n\nIN: ";
            // print2DVector(IN);

            // cout << "\n\nData:";
            // print2DVector(DA);
            // cout << "\n" << endl;

            clock_t t;
            t = clock();
   
            // conv_CPO_v9(O_v9, Kernel, IN_1d_v9,  DA_1d_v9, ptr_1d_v9, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
            conv_CPO_v9_trim(O_v9, Kernel, IN_1d_v9,  DA_1d_v9, ptr_1d_v9, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
            double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
            t_csr+=elapsed/(Ih*Iw*1.0); // normalized timing


            if(k == FREQ - 1)
            {
                cout << "Output V9: " << endl;
                print2DVectorF(O_v9);    
            }     
    }
  
  }  

    // You should call v9 here

    cout << "CPO conv time: " << t_csr << endl; 
    return 0;
}
