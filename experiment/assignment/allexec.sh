export OPENCV_INC=/usr/include/opencv
export OPENCV_LIB=/usr/lib/
export OPENCV_LIB_OPTS="-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann"

sudo apt-get update
sudo apt-get install -y gcc
sudo apt-get update
gcc  -I$OPENCV_INC codes/latest_affine.cpp -L$OPENCV_LIB $OPENCV_LIB_OPTS -o execs/affine.out
gcc  -I$OPENCV_INC codes/im_arith.cpp -L$OPENCV_LIB $OPENCV_LIB_OPTS -o execs/arith.out
gcc  -I$OPENCV_INC codes/im_dist.cpp -L$OPENCV_LIB $OPENCV_LIB_OPTS -o execs/dist.out
gcc  -I$OPENCV_INC codes/im_fourier.cpp -L$OPENCV_LIB $OPENCV_LIB_OPTS -o execs/fourier.out
gcc  -I$OPENCV_INC codes/im_hist.cpp -L$OPENCV_LIB $OPENCV_LIB_OPTS -o execs/hist.out
gcc  -I$OPENCV_INC codes/im_morphology.cpp -L$OPENCV_LIB $OPENCV_LIB_OPTS -o execs/morpho.out
gcc  -I$OPENCV_INC codes/im_nbrhood.cpp -L$OPENCV_LIB $OPENCV_LIB_OPTS -o execs/neigh.out 
gcc  -I$OPENCV_INC codes/im_path.cpp -L$OPENCV_LIB $OPENCV_LIB_OPTS -o execs/path.out
gcc  -I$OPENCV_INC codes/im_point.cpp -L$OPENCV_LIB $OPENCV_LIB_OPTS -o execs/point.out
gcc  -I$OPENCV_INC codes/im_segment.cpp -L$OPENCV_LIB $OPENCV_LIB_OPTS -o execs/segment.out
gcc  -I$OPENCV_INC codes/im_colour.cpp -L$OPENCV_LIB $OPENCV_LIB_OPTS -o execs/colour.out
gcc  -I$OPENCV_INC codes/im_colourLin.cpp -L$OPENCV_LIB $OPENCV_LIB_OPTS -o execs/colour2.out
gcc  -I$OPENCV_INC codes/im_colhist.cpp -L$OPENCV_LIB $OPENCV_LIB_OPTS -o execs/colour3.out
gcc  -I$OPENCV_INC codes/im_colhist.cpp -L$OPENCV_LIB $OPENCV_LIB_OPTS -o execs/colhist.out 
