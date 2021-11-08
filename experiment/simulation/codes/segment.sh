 gcc -I/usr/local/include/opencv -L/usr/local/lib -lcv -lhighgui -lstdc++ -o segment.out im_segment.cpp
mv segment.out ../execs/