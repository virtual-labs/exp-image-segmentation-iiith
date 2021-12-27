 gcc -I/home/intel/opencv/include/opencv -L/home/intel/opencv/lib -lcv -lhighgui -lstdc++ -o arith.out im_arith.cpp
 ./arith.out dst_1.jpg dst_2.jpg dst.jpg 3 2
 display dst_1.jpg
 display dst_2.jpg
 display dst.jpg

