#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include<iostream>
using namespace std;

int main( int argc, char** argv )

{
	IplImage *source = cvLoadImage(argv[1],0);

	cout<<"done this";

	IplImage *bin_img = cvCreateImage(cvSize( source->width, source->height ), CV_64FC1, 1 );
	bin_img = cvCloneImage(source);

	int c = cvGetSize(bin_img).width;
	int r = cvGetSize(bin_img).height;



	if(1)
	{
		IplImage *four_img = cvCreateImage(cvSize( source->width, source->height ),CV_64FC1,2);
		cvDFT( bin_img, four_img, CV_DXT_FORWARD, 0);
		IplImage *ioutRe = cvCreateImage(cvSize( source->width, source->height ),CV_64FC1, 1);
		IplImage *ioutIm = cvCreateImage(cvSize( source->width, source->height ),CV_64FC1, 1);
		cvSplit(four_img, ioutRe, ioutIm, NULL, NULL);
		cvSaveImage(argv[2],ioutRe);
		cvSaveImage(argv[3],ioutIm);
		cvReleaseImage( &ioutIm );
		cvReleaseImage( &ioutRe );
		cvReleaseImage( &source );
		cvReleaseImage( &four_img );
		cvReleaseImage( &bin_img );



	}

		return 0;
}
