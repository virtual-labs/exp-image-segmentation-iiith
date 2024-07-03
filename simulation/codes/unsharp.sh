 gcc -I/home/intel/opencv/include/opencv -L/home/intel/opencv/lib -lcv -lhighgui -lstdc++ -o unsharp.out im_unsharpmasking.cpp
 ./unsharp.out desert.jpg UnsharpMasking.jpg
display desert.jpg &
#display smoothened.jpg &
#display subtracted.jpg &
display UnsharpMasking.jpg &
