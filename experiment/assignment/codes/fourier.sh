 gcc -I/usr/local/include/opencv -L/usr/local/lib -lcv -lhighgui -lstdc++ -o fourier.out im_fourier.cpp
mv fourier.out ../execs/