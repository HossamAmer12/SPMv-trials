
 /usr/local/bin/gcc-6 -lc++ -O3 -march=native -ffast-math  -msse4 -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp $1
#/usr/local/bin/gcc-6 -lc++ -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp $1
#/usr/local/bin/gcc-6 -lc++ -O2 -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp $1

# If im2col encoding
#/usr/local/bin/gcc-6 -lc++ -O3 -march=native -ffast-math  -msse4 -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/ATLAS/include -fopenmp -lcblas -llapack $1

echo "/usr/local/bin/gcc-6 -lc++ -O3 -march=native -ffast-math  -msse4 -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp $1"
