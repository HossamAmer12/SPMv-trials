#include <iostream>
#include <vector>
#include <Eigen/Sparse>

using namespace Eigen;
using namespace std;


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


int main()
{
  int Ih = 5;
  int Iw = 5;
  SparseMatrix<double, RowMajor> sm(Ih, Iw);
  // SparseMatrix<double, ColMajor> sm(4,5);

  std::vector<int> cols = {0,1,4,0,4,0,4};
  std::vector<int> rows = {0,0,0,2,2,3,3};
  // std::vector<double> values = {0.2,0.4,0.6,0.3,0.7,0.9,0.2};
  std::vector<double> values = {1,1,1,1,1,1,1};

  for(int i=0; i < cols.size(); i++)
      sm.insert(rows[i], cols[i]) = values[i];
  
  sm.makeCompressed();

  // Prepare the Adata, Aindices, AindPtr for CSR multiplication
  int nz = sm.nonZeros();
  vector<double> Adata (sm.valuePtr(), sm.valuePtr() + nz);
  vector<int> Aindices (sm.innerIndexPtr(), sm.innerIndexPtr() + nz);
  vector<int> AindPtr (sm.outerIndexPtr(), sm.outerIndexPtr() + sm.outerSize()); // +1 for the last element

  std::cout << "\n===Sparse Feature Map: \n" << sm << std::endl;
  cout << "-----\n" << endl;
  std::cout << " Size of Ptr:  " << sm.outerSize() << endl;

  nz = sm.nonZeros();
  std::cout << "non_zeros : " << nz << " density: " << nz/(sm.size()*1.0) << std::endl;

  for (auto it = sm.valuePtr(); it != sm.valuePtr() + nz; ++it)
    std::cout << *it << std::endl;

  return 0;
}
