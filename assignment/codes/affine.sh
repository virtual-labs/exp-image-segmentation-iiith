 gcc -I/home/intel/opencv/include/opencv -L/home/intel/opencv/lib -lcv -lhighgui -lstdc++ -o affine.out latest_affine.cpp
 ./affine.out stock1.jpg dst.jpg 1 0 1 186 88 64 3 2 1
# display stock1.jpg
# display dst.jpg
