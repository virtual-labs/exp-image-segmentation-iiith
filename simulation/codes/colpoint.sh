 gcc -I/home/intel/opencv/include/opencv -L/home/intel/opencv/lib -lcv -lhighgui -lstdc++ -o colpoint.out im_colourLin.cpp
./colpoint.out colimg.jpg cimg.jpg 1 100 150 1 
#display colimg.jpg
#display cimg.jpg
