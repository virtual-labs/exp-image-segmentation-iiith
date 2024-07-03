 gcc -I/home/intel/opencv/include/opencv -L/home/intel/opencv/lib -lcv -lhighgui -lstdc++ -o dist.out im_dist.cpp
mv dist.out ../execs/
