#rm dst_1.jpg
gcc -I/home/intel/opencv/include/opencv -L/home/intel/opencv/lib -lcv -lhighgui -lstdc++ -o path.out im_path.cpp
gcc -I/home/intel/opencv/include/opencv -L/home/intel/opencv/lib -lcv -lhighgui -lstdc++ -o dist.out im_dist.cpp

#./path.out test.png dst_1.png 1 10 20 1 4
#./path.out kapil.jpg dst_1.jpg 150 150 160 175 4
# display dst_1.jpg
