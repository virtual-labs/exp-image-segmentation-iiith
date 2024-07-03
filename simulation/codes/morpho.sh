#rm dst_1.jpg
gcc -I/home/intel/opencv/include/opencv -L/home/intel/opencv/lib -lcv -lhighgui -lstdc++ -o morpho.out im_morphology.cpp
mv morpho.out ../morpho.out

