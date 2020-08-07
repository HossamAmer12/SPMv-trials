#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <stdlib.h>
#include <boost/timer/timer.hpp>
#include <iostream>

using namespace Eigen;
using namespace boost::timer;

typedef SparseMatrix<float> spMatFloat;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> deMatRowFloat;

void bench_Sparse(const spMatFloat &m, const deMatRowFloat &in, deMatRowFloat &o) {
  o.noalias()=m*in.transpose();
}

void bench_Dense(const deMatRowFloat &m, const deMatRowFloat &in, deMatRowFloat &o) {
  o.noalias()=m*in.transpose();
}

int main(int argc, const char **argv) {
  float ratio=1; // density
  int iter=20;
  int batch=32;
  float t_dense=0;
  float t_sparse=0;
  
  iter=1;
  batch=1;
  
   
  deMatRowFloat d_o1(batch,1000);
  deMatRowFloat d_o2(batch,1000);
  for(int k=0; k<iter; k++) {
    deMatRowFloat d_m=deMatRowFloat::Zero(1000,1000);
    deMatRowFloat d_b=deMatRowFloat::Random(batch,1000);
    for(int h=0;h<ratio*1000000;h++) {
      int i=rand()%1000;
      int j=rand()%1000;
      d_m(i,j)=(rand()%1000)/500.-1;
    }
    spMatFloat s_m=d_m.sparseView();
    {
      cpu_timer timer;
      for(int k=0;k<50;k++) bench_Dense(d_m,d_b,d_o1);
      cpu_times const elapsed_times(timer.elapsed());
      nanosecond_type const elapsed(elapsed_times.system+elapsed_times.user);
       t_dense+=elapsed/1000000.;
//	t_dense+=elapsed;
    }
    {
      cpu_timer timer;
      for(int k=0;k<50;k++) bench_Sparse(s_m,d_b,d_o2);
      cpu_times const elapsed_times(timer.elapsed());
      nanosecond_type const elapsed(elapsed_times.system+elapsed_times.user);
      t_sparse+=elapsed/1000000.;
      // t_sparse+=elapsed;
    }
  }
  std::cout<<"batch\t"<<batch<<"\tratio\t"<<ratio<<"\tdense\t"<<t_dense/50/iter<<"\tsparse\t"<<t_sparse/50/iter<<std::endl;
}
