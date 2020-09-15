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



void conv(MatrixXf& O, VectorXf& K, vector<vector<int> > &IN,  vector<vector<int> > &DA, vector<vector<int> > &ptr, int Kh, int Kw, int Oh, int Ow, int Sh, int Sw, int Ih, int Iw)
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
	   cout << "\nCurrent Submat " << submat << ", ptr:  " << ptr[type_ptr][submat] <<  ", ptr+1: " << ptr[type_ptr][submat+1]  << endl;
	  
	   for(int i = 0; i <= type_ptr; ++i)
	   {
	
	   // From ptr r t r+1
	   int shereet = (type_ptr > 0)? 1:0;
	   for(int x = ptr[type_ptr][submat-i*shereet]; x < ptr[type_ptr][submat-i*shereet+1]; ++x)
	   {
	     if(x == ptr[type_ptr][submat-i*shereet])
	     cout << "*********ptr_type: " << type_ptr <<  " shereet: " << shereet << " i: "  << i << " submat: " << submat << ", from: " << x << ", to: " << ptr[type_ptr][submat-i+1] << endl;
	    else
	     cout << "ptr_type: " << type_ptr <<  " shereet: " << shereet << " i: "  << i << " submat: " << submat << ", from: " << x << ", to: " << ptr[type_ptr][submat-i+1] << endl;
/**/	    
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
		
		    // O(y_out, x_out) += DA[type_ptr][x] * 1.0;
		    O(y_out, x_out) += DA[type_ptr][x] * K[input_index%Kw + l*Kw];
			
	     } // for each l in Kh

	    } // for each raga3 el shereet
	   } // for each x from range ptr and ptr + 1
	} // for each submat
	
	cout << "\n===CPO Output with Size: " << O.rows() << ", " << O.cols() <<  " \n" << O << std::endl;
        //if (type_ptr == 1)
	//exit(0);	
    }

    cout << "\n===CPO Output with Size: " << O.rows() << ", " << O.cols() <<  " \n" << O << std::endl;
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
 
    // Perform 50 times raw sparse matrix dense vector multiplication of CPO: d_o3 = d_m * d_b
    {
        clock_t t;
        t = clock();
        for(int k=0;k<1;k++) CPO(d_o3, filter_vectorized, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw,
				IN, DA, ptr);

	t = clock();
	conv(d_o1, filter_vectorized, IN,  DA, ptr, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
        //      for(int k=0;k<1;k++) _CPO(d_o3, filter_vectorized, org_fm, Kh, Kw, Oh, Ow, Sh, Sw, Ih, Iw);
        double elapsed = 1000*((double)(clock()-t))/CLOCKS_PER_SEC; // time in milliseconds
        t_csr+=elapsed/(Ih*Iw*1.0); // normalized timing
    }
   
    cout << "CPO conv time: " << t_csr << endl; 
    return 0;
}
