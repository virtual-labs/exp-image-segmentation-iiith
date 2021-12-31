 gcc -I/home/intel/opencv/include/opencv -L/home/intel/opencv/lib -lcv -lhighgui -lstdc++ -o coladaptivehist.out im_coladaptivehist.cpp
./coladaptivehist.out colimg.jpg cimg.jpg 7 7 2 2
#display cimg.jpg &
#display colimg.jpg &
