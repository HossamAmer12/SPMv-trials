#include <iostream>
#include <math.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/LU>

using namespace std;
using namespace Eigen;

int main() {

SparseMatrix<double, RowMajor> a(2,2);
SparseMatrix<double, ColMajor> b(2,2);
SparseMatrix<double, RowMajor> c(2,2);
a.coeffRef(0,0) = 1;
a.coeffRef(1,1) = 1;
b.coeffRef(0,0) = 9e-14;
b.coeffRef(1,1) = 1;
cout << "a" << endl;
cout << a << endl;
cout << "b" << endl;
cout << b << endl;
c = a * b;

cout << "c" << endl;
cout << c << endl;

//Matrix<double, Dynamic, Dynamic> a2(2,2);
//Matrix<double, Dynamic, Dynamic> b2(2,2);
//Matrix<double, Dynamic, Dynamic> c2(2,2);

MatrixXd a2 = MatrixXd::Zero(2, 2);
MatrixXd b2 = MatrixXd::Zero(2, 2);
MatrixXd c2 = MatrixXd::Zero(2, 2);

a2(0,0) = 1;
a2(1,1) = 1;
b2(0,0) = 9e-14;
b2(1,1) = 1;
cout << "a2" << endl;
cout << a2 << endl;
cout << "b2" << endl;
cout << b2 << endl;
c2 = a2 * b2;
cout << "c2" << endl;
cout << c2 << endl;

return 1;
}
