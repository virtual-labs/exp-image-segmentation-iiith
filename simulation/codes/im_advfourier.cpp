#include<iostream>
#include<algorithm>
#include <opencv/cxcore.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;

#define PI 3.1428571
#define MAXI 999999999

int main(int argc, char ** argv)
{
	IplImage * im = cvLoadImage(argv[1],0);
	IplImage * realInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
        IplImage * imaginaryInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
        IplImage * complexInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 2);
        CvMat* dft_A, tmp;
	int theta = atoi(argv[2]);
	int rad = atoi(argv[3]);
	int deltheta = atoi(argv[4]);
	int delrad = atoi(argv[5]);
	double angle1 = theta-(deltheta/2);
	double angle2 = theta+(deltheta/2);
	double angle3 = 180+theta-(deltheta/2);
	double angle4 = 180+theta+(deltheta/2);
	double val1 = atan((angle1==90||angle1==270||angle1==-90||angle1==-270)?MAXI:tan(PI/180*angle1));
	double val2 = atan((angle1==90||angle1==270||angle1==-90||angle1==-270)?MAXI:tan(PI/180*angle2));
	double val3 = atan((angle1==90||angle1==270||angle1==-90||angle1==-270)?MAXI:tan(PI/180*angle3));
	double val4 = atan((angle1==90||angle1==270||angle1==-90||angle1==-270)?MAXI:tan(PI/180*angle4));

	int dft_M, dft_N;
	cvScale(im, realInput, 1.0, 0.0);
        cvZero(imaginaryInput);
        cvMerge(realInput, imaginaryInput, NULL, NULL, complexInput);
        dft_M = cvGetOptimalDFTSize( im->height - 1 );
        dft_N = cvGetOptimalDFTSize( im->width - 1 );
        dft_A = cvCreateMat( dft_M, dft_N, CV_64FC2 );
	cvGetSubRect( dft_A, &tmp, cvRect(0,0, im->width, im->height));
        cvCopy( complexInput, &tmp, NULL );
        if( dft_A->cols > im->width )
        {
                cvGetSubRect( dft_A, &tmp, cvRect(im->width,0, dft_A->cols -
                                        im->width, im->height));
                cvZero( &tmp );
        }
        cvDFT( dft_A, dft_A, CV_DXT_FORWARD, complexInput->height );
	IplImage * re = cvCreateImage(cvSize(dft_N,dft_M),IPL_DEPTH_64F,1);
	IplImage * imag = cvCreateImage(cvSize(dft_N,dft_M),IPL_DEPTH_64F,1);
	cvSplit(dft_A,re,imag,0,0);

	int r = cvGetSize(imag).height;
	int c = cvGetSize(imag).width;
	double dist,row,col,val;
	CvScalar s;
	for (int i=0;i<r;i++)
	{
		for (int j=0;j<c;j++)
		{
			dist = sqrt(pow((i - r/2),2) + pow((j - c/2),2));
			if ((dist<=rad) && ((rad-delrad)<=dist))
			{
				if (deltheta==180)
				{
                                        s.val[0]=0.0;
                                        cvSet2D(imag,i,j,s);
				}
				else
				{
					row=i-r/2;
					col=j-c/2;
					if (col==0)
						val=atan(MAXI);
					else 
						val=atan(-row/col);
					if ((val1>=val2 && (val>=val1||val<=val2)) || (val3>=val4 && (val>=val3||val<=val4)))
					{
						s.val[0]=0.0;
						cvSet2D(imag,i,j,s);
					}
					else if ((val>=val1 && val<=val2) || (val>=val3 && val<=val4))
					{
						s.val[0]=0.0;
						cvSet2D(imag,i,j,s);
					}
				}
			}
		}
	}

	cvNamedWindow("imaginary",0);
	cvShowImage("imaginary",imag);

	cvMerge(re,imag,NULL,NULL,dft_A);
	cvDFT( dft_A, dft_A, CV_DXT_INVERSE_SCALE, dft_N);
        cvScale(dft_A,dft_A,0.001);
	cvSplit(dft_A,re,imag,0,0);
	cvPow(re,re,2);
	cvPow(imag,imag,2);
	cvAdd(re,imag,re,NULL);
	cvPow(re,re,0.5);
	cvAddS( re, cvScalarAll(1.0), re, NULL ); // 1 + Mag
        cvLog( re, re); // log(1 + Mag)
        double m,M;
        cvMinMaxLoc(re, &m, &M, NULL, NULL, NULL);
        cvScale(re, re, 1.0/(M-m), 1.0*(-m)/(M-m));
        IplImage * Output = cvCreateImage(cvGetSize(re),IPL_DEPTH_8U,1);
        CvPoint minLoc, maxLoc;
        double minVal = 0; double maxVal = 0;
        cvMinMaxLoc(re, &minVal, &maxVal, &minLoc, &maxLoc, 0);
        cvCvtScaleAbs(re,Output,255.0*(maxVal-minVal),0);
        cvSaveImage("advfourier.jpg",Output);

	cvNamedWindow("advfourier",0);
	cvShowImage("advfourier",Output);

	cvWaitKey(-1);

	cvReleaseImage(&im);
	cvReleaseImage(&realInput);
	cvReleaseImage(&imaginaryInput);
	cvReleaseImage(&complexInput);
	cvReleaseImage(&re);
	cvReleaseImage(&imag);
	cvReleaseMat(&dft_A);
	cvReleaseImage(&Output);
	return 0;
}
