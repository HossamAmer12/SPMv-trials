

#/usr/local/bin/gcc-6 -lc++ -o sparse -I/Users/hossam.amer/Library/Developer/Xcode/DerivedData/tf_conv-frxowusblodqdxcvddcqenieooto/Build/Intermediates/tf_conv.build/Debug/tf_conv.build/tf_conv-own-target-headers.hmap -I/Users/hossam.amer/Library/Developer/Xcode/DerivedData/tf_conv-frxowusblodqdxcvddcqenieooto/Build/Intermediates/tf_conv.build/Debug/tf_conv.build/tf_conv-all-target-headers.hmap -iquote /Users/hossam.amer/Library/Developer/Xcode/DerivedData/tf_conv-frxowusblodqdxcvddcqenieooto/Build/Intermediates/tf_conv.build/Debug/tf_conv.build/tf_conv-project-headers.hmap -I/Users/hossam.amer/Library/Developer/Xcode/DerivedData/tf_conv-frxowusblodqdxcvddcqenieooto/Build/Products/Debug/include -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -I/Users/hossam.amer/Library/Developer/Xcode/DerivedData/tf_conv-frxowusblodqdxcvddcqenieooto/Build/Intermediates/tf_conv.build/Debug/tf_conv.build/DerivedSources/x86_64 -I/Users/hossam.amer/Library/Developer/Xcode/DerivedData/tf_conv-frxowusblodqdxcvddcqenieooto/Build/Intermediates/tf_conv.build/Debug/tf_conv.build/DerivedSources -F/Users/hossam.amer/Library/Developer/Xcode/DerivedData/tf_conv-frxowusblodqdxcvddcqenieooto/Build/Products/Debug -MMD -MT dependencies -MF /Users/hossam.amer/Library/Developer/Xcode/DerivedData/tf_conv-frxowusblodqdxcvddcqenieooto/Build/Intermediates/tf_conv.build/Debug/tf_conv.build/Objects-normal/x86_64/main.d -c /Users/hossam.amer/7aS7aS_Works/work/my_Tools/tf_conv/tf_conv/main.cpp 

#/Users/hossam.amer/Library/Developer/Xcode/DerivedData/tf_conv-frxowusblodqdxcvddcqenieooto/Build/Intermediates/tf_conv.build/Debug/tf_conv.build/Objects-normal/x86_64/main.o 

#/usr/local/bin/gcc-6 -lc++ -o hello tf_conv/main.cpp 


#/usr/local/bin/gcc-6 -lc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen  -c e.cpp

#/usr/local/bin/gcc-6 -lc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp e.cpp
#/usr/local/bin/gcc-6 -lc++ -w -Wignored-attributes -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp conv.cpp

# non optimized:
#/usr/local/bin/gcc-6 -lc++ -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp conv.cpp

# Optimized
#/usr/local/bin/gcc-6 -lc++ -O2 -msse4 -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp conv.cpp
#/usr/local/bin/gcc-6 -lc++ -O2 -msse4 -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp conv_asl.cpp

# Compile conv.cpp 
# -march=haswell
#/usr/local/bin/gcc-6 -lc++ -O3 -march=native -ffast-math  -msse4 -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp conv.cpp

# Compile dummy
#/usr/local/bin/gcc-6 -lc++ -O3 -march=native -ffast-math  -msse4 -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp dummy.cpp

# Compile lower.cpp
# /usr/local/bin/gcc-6 -lc++ -O3 -march=native -ffast-math  -msse4 -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp lower.cpp

# Compile loop_conv.cpp
#/usr/local/bin/gcc-6 -lc++ -O3 -march=native -ffast-math  -msse4 -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp loop_conv.cpp

# Enable -O2 
# Enable NDEBUG
# Enable vectorization
# Enable OMP

# Compile quick trials
#/usr/local/bin/gcc-6 -lc++ -O3 -march=native -ffast-math  -msse4 -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp back.cpp
#/usr/local/bin/gcc-6 -lc++ -O3 -march=native -ffast-math  -msse4 -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp test_mult.cpp

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# module load eigen/3.2.10
# module load gcc/6.4.0
# module load boost/1.72.0

# server 
# Compile lower.cpp
# gcc -lc++ -O3 -march=native -ffast-math  -msse4 -w -lboost_timer -lstdc++ -o sparse -I/Users/hossam.amer/7aS7aS_Works/work/my_Tools/eigen -fopenmp lower.cpp

# gcc -O3 -march=native -ffast-math  -msse4 -w -lstdc++ -o sparse -fopenmp lower.cpp

gcc -O3 -march=native -ffast-math  -msse4 -w -lstdc++ -o loop_conv -fopenmp loop_conv.cpp


# g++ -O3 -fopenmp gs_test.cpp -o gs_test

time ./loop_conv