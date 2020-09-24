#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <boost/timer/timer.hpp>
#include <time.h>
#include <fstream>


#define IS_PRINT        0
#define IS_PRINT_SIZE   0

using namespace Eigen;
using namespace std;
using namespace boost::timer;


void print2DVector(std::vector<vector<int>>& x);
void print2DVectorF(std::vector<vector<float>>& x);
void printVector(std::vector<int>& x);


// https://scicomp.stackexchange.com/questions/27977/how-can-i-speed-up-this-code-for-sparse-matrix-vector-multiplication

bool isEqualVectors(std::vector<vector<float>>& x1, std::vector<vector<float>>& x2)
{
    bool error = false;

    assert(x1.size() == x2.size());
    assert(x1[0].size() == x2[0].size());
    for(int i = 0; i < x1.size(); ++i)
    {
        for(int j = 0; j < x1[i].size(); ++j)
        {
            if(x1[i][j] != x2[i][j])
            {
              cout << "Error! " << i << ", " << j << endl;
              error = true;
              break;

            }
        }

        if(error)
          break;
    }
    
  return error;
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
        for(int j = 0; j < x[i].size(); ++j)
        {
            cout << x[i][j] << " ";
        }
        cout << "\n";
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

void reset2DVectorF(std::vector<vector<float>>& x)
{
    for(int i = 0; i < x.size(); ++i)
    {
        for(int j = 0; j < x[i].size(); ++j)
        {
            x[i][j] = 0;
        }
    }
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
        if(m < 0 || m >= Oh) continue;
                 
        // cout << "R) " << m << ", C) " << n << ", " << Kindex << endl;
        O[m][n] += NZE_data*K[Kindex];
      }   
    }   

  }
} // end mult



/*SSSSSSSSSSSSSSSSSSSSSSSSSSS*/

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

// Removing repeats + previous optimizations
void conv_CPO_v7(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<int> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
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

      // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;

   
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

    // cout << "Output: " << endl;
    // print2DVectorF(O);
    // cout << "-----\n" << endl;
}



void conv_CPO_v7_trim(vector<vector<float> > & O, vector<int> const &K, vector<int>  &IN,  vector<int> &DA, vector<int>  &ptr, const int Kh, const int Kw, const int Oh, const int Ow, const int Sh, const int Sw, const int Ih, const int Iw)
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
    {
     
      x              = *x_ptr; 
      int end_x_loop = *(x_ptr+1); 
      ++x_ptr;

      // cout << "Sumbat: " << submat << ", type_ptr: " << type_ptr << " ind to fetch: " << type_ptr*number + submat + 1 << " start " << x  << " end: " << end_x_loop  << endl;
      // cout << "V5) Submat: " << submat << ", type_ptr: " << type_ptr << " start " << x  << " end: " << end_x_loop  << endl;

   
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
            
            bool flag = true;

            if(y_out >= 0 && y_out < Oh) {
               O[y_out][x_out] += used_data * K[input_index%Kw + l*Kw];

               for(int i = 1; i <= type_ptr; ++i)
               {
                 int input_index     = used_index - i;
                 O[y_out][x_out + i] += used_data * K[input_index%Kw + l*Kw]; 

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


int main()
{
  
  // density:
  // float density = 0.05;
  // float density = 1; 
  float density = 0.5;
   



  // std::vector<int> I_list = {8,17,50};
  // std::vector<int> Kh_list = {3,7, 1};
  // std::vector<int> Kw_list = {3,1, 7};

  std::vector<int> I_list = {8};
  std::vector<int> Kh_list = {3};
  std::vector<int> Kw_list = {3};


  // std::vector<int> I_list = {17};
  // std::vector<int> Kh_list = {7};
  // std::vector<int> Kw_list = {1};



 for(; density < 1.05; density+=0.05)
  {  


  // timer for im2col, csr
  float t_im2col = 0;
  float t_csr    = 0;
  float t_cpo    = 0;
  float t_cpo1    = 0;

  // bench iterations
 // int bench_iterations = 100000;

  // int bench_iterations = 1;
  // int bench_iterations = 2;
  int bench_iterations = 1;
   // int bench_iterations = 100;  

  // Conv parameters:
  int padding = 0;
  int stride  = 1;
  int Sh, Sw; 
  Sh = Sw = stride;
  int num_filters = 1; // 64


  for(int I: I_list)
  {
    int Ih = I;
    int Iw = I;
  
   for(int KK = 0; KK < Kh_list.size(); ++KK)
  {


    int Kh = Kh_list[KK];
    int Kw = Kw_list[KK];
 
    // usleess case skip it
    if(Ih == 8)
      if(Kh == 1 || Kh == 7)
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
      
  
  // Create the filter K and its vectorized version:
  MatrixXf filter             = MatrixXf::Ones(Kh, Kw);
  VectorXf filter_vectorized  = VectorXf::Ones(Kh*Kw);

  // std::cout << "\n===Filter: " <<  " \n" << filter_vectorized  << std::endl;
  // cout << "-----\n" << endl;

  // Prepare the output for im2col, sparseMat
  MatrixXf d_o1 = MatrixXf::Zero(Oh, Ow);
  MatrixXf d_o2 = MatrixXf::Zero(Oh, Ow);
  
  // Create the Kernel
  vector<int> Kernel(Kh*Kw, 1);
  Kernel[2] = 2;

  

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

    // CPO with repeats until v5
    // transform 2d to 1d:
    int count_ptr = 0;
    for(int i = 0 ; i < ptr.size(); ++i)
    {
      count_ptr += ptr[i].size();
    }

    int count_d = 0;
    for(int i = 0 ; i < DA.size(); ++i)
    {
      count_d += DA[i].size();
    }

    std::vector<int> IN_1d(count_d, 0);
    std::vector<int> DA_1d(count_d, 0);
    std::vector<int> ptr_1d(count_ptr, 0);

    transform2dTo1d(IN, DA, ptr, IN_1d, DA_1d, ptr_1d);
    ///////////


    // Perform 50 times raw sparse matrix dense vector multiplication: d_CPO = d_m * d_b

    vector<vector<float> > O( Oh , vector<float> (Ow, 0));
    { 

        clock_t t;

      // cout << ptr.size() << "\t" << ptr[0].size() << endl;
       for(int k=0;k<bench_iterations;k++){

           // Prepare the output for CPO           
           t = clock();
           // conv_CPO(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           // conv_CPO_v1(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           // conv_CPO_v2(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           // conv_CPO_v3(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

           conv_CPO_v5(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
           t_cpo+=elapsed/(Ih*Iw*1.0); // normalized timing

           // if(k == bench_iterations-1)
           // {
           //  cout << "CPO V5: " << endl;
           //  print2DVectorF(O);
           //  cout << "-----\n" << endl;
           // }

           // Reset in all except last
           if(k != bench_iterations - 1)
            reset2DVectorF(O);
           
       }
    }


      // CPO without repeats v7 onwards
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
    ///////////


  // Perform 50 times raw sparse matrix dense vector multiplication: d_CPO = d_m * d_b

       // cout << "Org Ptr: " << endl;
       // print2DVector(ptr);
       // cout << "V5 Ptr: " << endl;
       // printVector(ptr_1d);
       // cout << "V7 Ptr: " << endl;
       // printVector(ptr_1d_v7);
       // exit(0);

    vector<vector<float> > O1( Oh , vector<float> (Ow, 0));
   {  

      if(n != 1)
      {
        clock_t t;
        
    
      // cout << ptr.size() << "\t" << ptr[0].size() << endl;
       for(int k=0;k<bench_iterations;k++){

           // Prepare the output for CPO           
           t = clock();
           // conv_CPO(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           // conv_CPO_v1(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           // conv_CPO_v2(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           // conv_CPO_v3(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

           conv_CPO_v7_trim(O1, Kernel, IN_1d_v7,  DA_1d_v7, ptr_1d_v7, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
           t_cpo1+=elapsed/(Ih*Iw*1.0); // normalized timing


           // if(k == bench_iterations-1)
           // {
           //  cout << "CPO V7: " << endl;
           //  print2DVectorF(O1);
           //  cout << "-----\n" << endl;
           // }

           // Reset in all except last
           if(k != bench_iterations - 1)
            reset2DVectorF(O1);

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
       for(int k=0;k<bench_iterations;k++){

           // Prepare the output for CPO           
           t = clock();
           // conv_CPO(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           // conv_CPO_v1(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           // conv_CPO_v2(O, Kernel, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           // conv_CPO_v3(O, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);

           conv_CPO_v7(O1, Kernel, IN_1d,  DA_1d, ptr_1d, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
           double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
           t_cpo1+=elapsed/(Ih*Iw*1.0); // normalized timing


           // if(k == bench_iterations-1)
           // {
           //  cout << "CPO V7: " << endl;
           //  print2DVectorF(O1);
           //  cout << "-----\n" << endl;
           // }

           // Reset in all except last
           if(k != bench_iterations - 1)
            reset2DVectorF(O1);

           // if(k == bench_iterations-1)
           // {
           //  print2DVectorF(O);
           //  cout << "-----\n" << endl;
           // }
       }
     }
   }  

   // CSCC
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

  // Create the sparse representation of the lowered matrix:
  SparseMatrix<float, RowMajor> lowered_mat_sparse = lowered_mat.sparseView();
  lowered_mat_sparse.makeCompressed();
 
  // Prepare the Adata, Aindices, AindPtr for CSR multiplication
  int nz = lowered_mat_sparse.nonZeros();
  vector<double> Adata (lowered_mat_sparse.valuePtr(), lowered_mat_sparse.valuePtr() + nz);
  vector<int> Aindices (lowered_mat_sparse.innerIndexPtr(), lowered_mat_sparse.innerIndexPtr() + nz);
  vector<int> Aindptr (lowered_mat_sparse.outerIndexPtr(), lowered_mat_sparse.outerIndexPtr() + lowered_mat_sparse.outerSize()); // +1 for the last element
  // push back the last element the number of nnz in ptr:
  Aindptr.push_back(nz);


   // Perform 50 times raw sparse matrix dense vector multiplication: d_o2 = d_m * d_b [Without Eigen]
  vector<vector<float> > O_CSR( Oh , vector<float> (Ow, 0));
  {  
      clock_t t;
       for(int k=0;k<bench_iterations;k++){
           // Prepare the output for CSR
           t = clock();
           // csrMult_v2(d_o2, filter_vectorized, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
           csrMult_v4(O_CSR, Kernel, Adata, Aindices, Aindptr, Kh, Kw, Oh, Ow);
           double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds 
           t_csr+=elapsed/(Ih*Iw*1.0); // normalized timing


           // if(k == bench_iterations - 1)
           // {
           //    cout << "CSR without Eigen Output: " << endl;
           //    print2DVectorF(O_CSR);
           //    cout << "-----\n" << endl;
           // }


           if(k != bench_iterations - 1)
            reset2DVectorF(O_CSR);

       }
   }

  


    // bool error2 = isEqualVectors(O, O1);
    // if(!error2)
    //   std::cout << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batchCPO\t"<<1
    //     <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
    //     <<"\tverify:\t"<<std::boolalpha<<!error2<<"\n";
    // else
    //     std::cout << "XXXXXXXXXXX\t" << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batchCPO\t"<<1
    //     <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
    //     <<"\tverify:\t"<<std::boolalpha<<!error2<<"\n";    



    // bool error = isEqualVectors(O_CSR, O);
  bool error = isEqualVectors(O_CSR, O1);


   // print2DVectorF(O);
   // print2DVectorF(O_CSR);
   if(!error)
      std::cout << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batchCSR\t"<<1
        <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
        <<"\tverify:\t"<<std::boolalpha<<!error  << "\tim2col\t" << t_im2col << "\tcsr\t" << t_csr << "\tcpoV5\t" << t_cpo << "\tcpoV7\t" << t_cpo1 
        << "\t S2: " << std::boolalpha <<  (t_cpo1 <= t_csr)   <<"\n";
    else
        std::cout << "XXXXXXXXXXX\t" << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batchCSR\t"<<1
        <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
        <<"\tverify:\t"<<std::boolalpha<<!error<<"\n";

  // cout << "im2col t: " << t_im2col << " csr t: " << t_csr << " cpoV5 t: " << t_cpo << " cpoV7 t: " << t_cpo1 << endl;
  // elapsed time per feature element in the entire bench iterations
  // std::cout<<"batch\t"<<In<<"\tdensity\t"<<density <<"\tdensity_cal\t"<<density_cal <<"\tim2col\t"<< t_im2col <<"\tcsr\t"<< t_csr <<"\tcpo\t"<< t_cpo <<std::endl;
    // std::cout << "CPO:\t" << Kh << "x" << Kw  << " | " <<  Ih << "x" << Iw <<  ") batch\t"<<1
    //     <<"\ttarget_density\t"<<density<<"\tdensity\t"<<density_cal
    //     <<"\tim2col\t"<<t_im2col<<"\tcsr\t" <<t_csr <<"\tcpo\t"<< t_cpo
    //      <<"\tpercent1\t"<< 100.0*(t_im2col-t_csr)/t_im2col  <<"\tpercent2\t"<< 100.0*(t_im2col-t_cpo)/t_im2col << "\n";

  
  } // end K list loop
  } // end I list loop
  

  } // density loop

  cout << "Smile All Done!"<< endl;
 
  return 0;
}
