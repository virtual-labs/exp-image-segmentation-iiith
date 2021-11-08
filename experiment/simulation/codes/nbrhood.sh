#rm dst_1.jpg
gcc -I/home/intel/opencv/include/opencv -L/home/intel/opencv/lib -lcv -lhighgui -lstdc++ -o nbrhood.out im_nbrhood.cpp
mv nbrhood.out ../neigh.out

