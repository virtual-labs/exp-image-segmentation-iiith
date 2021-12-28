#include<iostream>
#include<algorithm>
#include <opencv/cxcore.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;

int main(int argc, char ** argv)
{
	IplImage * magImage = cvLoadImage(argv[1],0);
	IplImage * phaseImage = cvLoadImage(argv[2],0);
	IplImage * realInput = cvCreateImage( cvGetSize(magImage), IPL_DEPTH_64F, 1);
        IplImage * imaginaryInput = cvCreateImage( cvGetSize(magImage), IPL_DEPTH_64F, 1);
        IplImage * complexInput = cvCreateImage( cvGetSize(magImage), IPL_DEPTH_64F, 2);

	cvScale(magImage, realInput, 1.0, 0.0);
        cvZero(imaginaryInput);
        cvMerge(realInput, imaginaryInput, NULL, NULL, complexInput);
	int dft_M = cvGetOptimalDFTSize( magImage->height - 1 );
        int dft_N = cvGetOptimalDFTSize( magImage->width - 1 );
	CvMat* dft_A, tmp;
	dft_A = cvCreateMat( dft_M, dft_N, CV_64FC2 );

	cvGetSubRect( dft_A, &tmp, cvRect(0,0, magImage->width, magImage->height));
        cvCopy( complexInput, &tmp, NULL );
        if( dft_A->cols > magImage->width )
        {
                cvGetSubRect( dft_A, &tmp, cvRect(magImage->width,0, dft_A->cols -
                                        magImage->width, magImage->height));
                cvZero( &tmp );
        }

        cvDFT( dft_A, dft_A, CV_DXT_FORWARD, complexInput->height );
	IplImage * re = cvCreateImage( cvSize(dft_N,dft_M), IPL_DEPTH_64F, 1);
	IplImage * im = cvCreateImage( cvSize(dft_N,dft_M), IPL_DEPTH_64F, 1);
	IplImage * mag = cvCreateImage( cvSize(dft_N,dft_M), IPL_DEPTH_64F, 1);
	cvSplit( dft_A, re, im, 0, 0 );
	cvPow( re, re, 2.0);
        cvPow( im, im, 2.0);
        cvAdd( re, im, mag, NULL);
        cvPow( mag, mag, 0.5 );

	cvScale(phaseImage, realInput, 1.0, 0.0);
        cvZero(imaginaryInput);
        cvMerge(realInput, imaginaryInput, NULL, NULL, complexInput);
        //int dft_M = cvGetOptimalDFTSize( im->height - 1 );
        //int dft_N = cvGetOptimalDFTSize( im->width - 1 );
        //cvMat* dft_A = cvCreateMat( dft_M, dft_N, CV_64FC2 );
        //cvMat tmp;
        cvGetSubRect( dft_A, &tmp, cvRect(0,0, phaseImage->width, phaseImage->height));
        cvCopy( complexInput, &tmp, NULL );
        if( dft_A->cols > phaseImage->width )
        {
                cvGetSubRect( dft_A, &tmp, cvRect(phaseImage->width,0, dft_A->cols -
                                        phaseImage->width, phaseImage->height));
                cvZero( &tmp );
        }
        cvDFT( dft_A, dft_A, CV_DXT_FORWARD, complexInput->height );
        cvSplit( dft_A, re, im, 0, 0 );


	IplImage *realOutput = cvCreateImage(cvSize(dft_N,dft_M),IPL_DEPTH_64F,1);
	IplImage *imagOutput = cvCreateImage(cvSize(dft_N,dft_M),IPL_DEPTH_64F,1);
	IplImage *complexOutput = cvCreateImage(cvSize(dft_N,dft_M),IPL_DEPTH_64F,2);
	int r=cvGetSize(mag).height;
	int c=cvGetSize(mag).width;
	CvScalar s1,s2,s3,s4;
	for (int i=0;i<r;i++)
	{
		for (int j=0;j<c;j++)
		{
			s1=cvGet2D(mag,i,j);
			s2=cvGet2D(re,i,j);
			s3=cvGet2D(im,i,j);
			s4.val[0]=(s1.val[0]*s2.val[0])/(sqrt((s2.val[0]*s2.val[0]) + (s3.val[0]*s3.val[0])));
			cvSet2D(realOutput,i,j,s4);
			s4.val[0]=(s1.val[0]*s3.val[0])/(sqrt((s2.val[0]*s2.val[0]) + (s3.val[0]*s3.val[0])));
			cvSet2D(imagOutput,i,j,s4);
		}
	}
	cvMerge(realOutput, imagOutput, NULL, NULL, dft_A);
	cvDFT( dft_A, dft_A, CV_DXT_INVERSE_SCALE, dft_N);//complexInput->height );
        cvScale(dft_A,dft_A,0.001);
	cvSplit( dft_A, realOutput, imagOutput, 0, 0 );
        cvPow( realOutput, realOutput, 2.0);
        cvPow( imagOutput, imagOutput, 2.0);
        cvAdd( realOutput, imagOutput, realOutput, NULL);
        cvPow( realOutput, realOutput, 0.5 );
        cvAddS( realOutput, cvScalarAll(1.0), realOutput, NULL ); // 1 + Mag
        cvLog( realOutput, realOutput ); // log(1 + Mag)
	double m,M;
        cvMinMaxLoc(realOutput, &m, &M, NULL, NULL, NULL);
        cvScale(realOutput, realOutput, 1.0/(M-m), 1.0*(-m)/(M-m));
	IplImage * Output = cvCreateImage(cvGetSize(realOutput),IPL_DEPTH_8U,1);
	CvPoint minLoc, maxLoc;
        double minVal = 0; double maxVal = 0;
        cvMinMaxLoc(realOutput, &minVal, &maxVal, &minLoc, &maxLoc, 0);
        cvCvtScaleAbs(realOutput,Output,255.0*(maxVal-minVal),0);
        cvSaveImage("impofphase.jpg",Output);


	cvReleaseImage(&re);
	cvReleaseImage(&im);
	cvReleaseImage(&mag);
	cvReleaseImage(&magImage);
	cvReleaseImage(&phaseImage);
	cvReleaseImage(&realInput);
	cvReleaseImage(&imaginaryInput);
	cvReleaseImage(&complexInput);
	cvReleaseImage(&realOutput);
	cvReleaseImage(&imagOutput);
	cvReleaseImage(&complexOutput);
	cvReleaseImage(&Output);
	cvReleaseMat(&dft_A);

	return 0;
}
