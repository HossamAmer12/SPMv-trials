import glob, os
import numpy as np
# Gen the coo files


# All stride of 1

# Ih = Iw = np.array([149 , 147 , 47 , #mixed0 conv 2,3,5 
# 					35 , #mixed 1 conv3 
# 					35 , #mixed 2 conv6 
# 					17 , 17 , #mixed 5 conv3,4
# 					17 , 17 , #mixed 8 conv8,9 
# 					17 , 17 , #mixed 9 conv 4,5 
# 					])

# Ic		= np.array([


# density = np.array([0.77 , 0.58 , 0.61 ,
# 					0.63 ,
# 					0.40 ,
# 					0.23 , 0.35,
# 					0.17 , 0.20,
# 					0.22 , 0.26,
# 					])
# Kh 		= np.array([3 , 3 , 3 ,
# 					5 ,
# 					3 ,
# 					1 , 7,
# 					7 , 1,
# 					1 , 7,
# 					])
# Kw 		= np.array([3 , 3 , 3 ,
# 					5 ,
# 					3 ,
# 					7 , 1,
# 					1 , 7,
# 					7 , 1,
# 					])


Ih = Iw = np.array([149 , 147 , 73 , #ID 1 2 3
					 71 , 35  , 35 , #ID 4 5 7 
					 17 , 17  , 17 , #ID 52 53 54
					 8  , 			 #ID 89
					 8 ,  8   , 8  , #ID 91 92 93
					])

Ic		= np.array([32 , 32  , 64 ,
					80 , 192 , 48 ,
					160, 160 , 768,
					2048,
					384 , 384 , 2048,
					])

density = np.array([0.59 , 0.46 , 0.77 ,
					0.48 , 0.74 , 0.48 ,
					0.14 , 0.17 , 0.24 ,
					0.08 ,
					0.14 , 0.14, 0.24,
					])

Kh 		= np.array([3 , 3 , 1 ,
					3 , 1 , 5 ,
					1 , 7 , 1 ,
					1 , 
					1 , 3 , 1,
					])

Kw 		= np.array([3 , 3 , 1 ,
					3 , 1 , 5 ,
					7 , 1 , 1 ,
					1 , 
					3 , 1 , 1 ,
					])

# Ih = Iw = np.array([ 35 ,  
# 					])

# Ic		= np.array([ 64 ,
# 					])

# density = np.array([ 0.26, 
# 					])

# Kh 		= np.array([3 ,
# 					])

# Kw 		= np.array([3 ,
# 					])

# generate the coo file & cal the time 
# ./loop_conv_last 0.05 149 149 3 3 100
bench_iterations = 1000
dir_mtx = "/scratch/ahamsala/DC_CODES/SPMv-trials"
for density in np.arange(0.05, 1.05, 0.05):
	for i in range(0, len(Ih)):
	    os.system("cd /scratch/ahamsala/DC_CODES/SPMv-trials/; \
	    		   ./loop_conv_last %.2f %d %d %d %d %d  " % (density, Ih[i], Iw[i]*Ic[i], Kh[i], Kw[i], bench_iterations))


# run MLK or merg 
dir_mtx = "/scratch/ahamsala/DC_CODES/OUTPUTs/COO"
os.chdir(dir_mtx)

# ./cpu_spmv --mtx=/scratch/ahamsala/DC_CODES/OUTPUTs/COO/coo_149_0.10.mtx

for file in glob.glob("*.mtx"):
    file_name = os.path.join(dir_mtx, file)
    os.system("cd /home/ahamsala/scratch/DC_CODES/merge-spmv/; \
    		   ./cpu_spmv --mtx=%s " % file_name)



