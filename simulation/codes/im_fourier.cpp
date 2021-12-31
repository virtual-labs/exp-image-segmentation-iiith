#include<iostream>
#include<algorithm>
//#include "stdafx.h"
#include <opencv/cxcore.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>

#define PI 3.1428571
#define MAXI 999999999

// Rearrange the quadrants of Fourier image so that the origin is at
// the image center
// src & dst arrays of equal size & type
void cvShiftDFT(CvArr * src_arr, CvArr * dst_arr )
{
	CvMat * tmp;
	CvMat q1stub, q2stub;
	CvMat q3stub, q4stub;
	CvMat d1stub, d2stub;
	CvMat d3stub, d4stub;
	CvMat * q1, * q2, * q3, * q4;
	CvMat * d1, * d2, * d3, * d4;

	CvSize size = cvGetSize(src_arr);
	CvSize dst_size = cvGetSize(dst_arr);
	int cx, cy;

	if(src_arr==dst_arr){
		tmp = cvCreateMat(size.height/2, size.width/2,
				cvGetElemType(src_arr));
	}

	cx = size.width/2;
	cy = size.height/2; // image center

	q1 = cvGetSubRect( src_arr, &q1stub, cvRect(0,0,cx, cy) );
	q2 = cvGetSubRect( src_arr, &q2stub, cvRect(cx,0,cx,cy) );
	q3 = cvGetSubRect( src_arr, &q3stub, cvRect(cx,cy,cx,cy) );
	q4 = cvGetSubRect( src_arr, &q4stub, cvRect(0,cy,cx,cy) );
	d1 = cvGetSubRect( src_arr, &d1stub, cvRect(0,0,cx,cy) );
	d2 = cvGetSubRect( src_arr, &d2stub, cvRect(cx,0,cx,cy) );
	d3 = cvGetSubRect( src_arr, &d3stub, cvRect(cx,cy,cx,cy) );
	d4 = cvGetSubRect( src_arr, &d4stub, cvRect(0,cy,cx,cy) );

	if(src_arr!=dst_arr){
		cvCopy(q3, d1, 0);
		cvCopy(q4, d2, 0);
		cvCopy(q1, d3, 0);
		cvCopy(q2, d4, 0);
	}
	else{
		cvCopy(q3, tmp, 0);
		cvCopy(q1, q3, 0);
		cvCopy(tmp, q1, 0);
		cvCopy(q4, tmp, 0);
		cvCopy(q2, q4, 0);
		cvCopy(tmp, q2, 0);
	}
}

int main(int argc, char ** argv)
{
	if(atoi(argv[1])==1)
	{
		IplImage * im;
		IplImage * im1;
		IplImage * realInput;
		IplImage * imaginaryInput;
		IplImage * complexInput;

		int dft_M, dft_N;
		CvMat* dft_A, tmp;
		IplImage * image_Re;
		IplImage * image_Im;
		IplImage * image_Re2;
		IplImage * image_Im2;
		double m, M;

		im1 = cvLoadImage( argv[2],0 );
		im = cvLoadImage( argv[2], CV_LOAD_IMAGE_GRAYSCALE );
		if( !im )
			return -1;

		realInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
		imaginaryInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
		complexInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 2);

		cvScale(im, realInput, 1.0, 0.0);
		cvZero(imaginaryInput);
		cvMerge(realInput, imaginaryInput, NULL, NULL, complexInput);

		dft_M = cvGetOptimalDFTSize( im->height - 1 );
		dft_N = cvGetOptimalDFTSize( im->width - 1 );

		dft_A = cvCreateMat( dft_M, dft_N, CV_64FC2 );
		image_Re = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
		image_Im = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
		image_Re2 = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
		image_Im2 = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);

		// copy A to dft_A and pad dft_A with zeros
		cvGetSubRect( dft_A, &tmp, cvRect(0,0, im->width, im->height));
		cvCopy( complexInput, &tmp, NULL );
		if( dft_A->cols > im->width )
		{
			cvGetSubRect( dft_A, &tmp, cvRect(im->width,0, dft_A->cols -
						im->width, im->height));
			cvZero( &tmp );
		}


		cvDFT( dft_A, dft_A, CV_DXT_FORWARD, complexInput->height );

		// Split Fourier in real and imaginary parts
		cvSplit( dft_A, image_Re, image_Im, 0, 0 );
		cvCopyImage(image_Re,image_Re2);
		cvCopyImage(image_Im,image_Im2);

		// Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
		cvPow( image_Re, image_Re, 2.0);
		cvPow( image_Im, image_Im, 2.0);
		cvAdd( image_Re, image_Im, image_Re, NULL);
		cvPow( image_Re, image_Re, 0.5 );

		// compute the phase of the spectrum Phase = atan(Re/Im)
		int r=cvGetSize(image_Re).height;
		int c=cvGetSize(image_Re).width;
		CvScalar s1,s2;
		for (int i=0;i<r;i++)
		{
			for (int j=0;j<c;j++)
			{
				s1=cvGet2D(image_Re2,i,j);
				s2=cvGet2D(image_Im2,i,j);
				s1.val[0]=atan((s2.val[0]/s1.val[0]));
				cvSet2D(image_Re2,i,j,s1);
			}
		}

		// Compute log(1 + Mag)
		cvAddS( image_Re, cvScalarAll(1.0), image_Re, NULL ); // 1 + Mag
		cvLog( image_Re, image_Re ); // log(1 + Mag)

		// Rearrange the quadrants of Fourier image so that the origin is at
		// the image center
		cvShiftDFT( image_Re, image_Re );

		cvMinMaxLoc(image_Re, &m, &M, NULL, NULL, NULL);
		cvScale(image_Re, image_Re, 1.0/(M-m), 1.0*(-m)/(M-m));

		IplImage *image_Re1 = cvCreateImage( cvGetSize(image_Re), IPL_DEPTH_8U, 1);
		CvPoint minLoc, maxLoc;
		double minVal = 0, maxVal = 0;
		cvMinMaxLoc(image_Re, &minVal, &maxVal, &minLoc, &maxLoc, 0);
		cvCvtScaleAbs(image_Re,image_Re1,255.0*(maxVal-minVal),0);
		cvSaveImage(argv[3],image_Re1);
		//cvNamedWindow("magnitude",0);
		//cvShowImage("magnitude",image_Re1);

		cvMinMaxLoc(image_Re2, &m, &M, NULL, NULL, NULL);
		cvScale(image_Re2, image_Re2, 1.0/(M-m), 1.0*(-m)/(M-m));

		minVal = 0, maxVal = 0;
		cvMinMaxLoc(image_Re2, &minVal, &maxVal, &minLoc, &maxLoc, 0);
		cvCvtScaleAbs(image_Re2,image_Re1,255.0*(maxVal-minVal),0);
		cvSaveImage(argv[4],image_Re1);
		//cvNamedWindow("phase",0);
		//cvShowImage("phase",image_Re1);

		//FOR INVERSE*******
		cvDFT( dft_A, dft_A, CV_DXT_INVERSE_SCALE, dft_N);//complexInput->height );
		cvScale(dft_A,dft_A,0.001);

		// Split Fourier in real and imaginary parts
		cvSplit( dft_A, image_Re, image_Im, 0, 0 );

		// Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
		cvPow( image_Re, image_Re, 2.0);
		cvPow( image_Im, image_Im, 2.0);
		cvAdd( image_Re, image_Im, image_Re, NULL);
		cvPow( image_Re, image_Re, 0.5 );

		// Compute log(1 + Mag)
		cvAddS( image_Re, cvScalarAll(1.0), image_Re, NULL ); // 1 + Mag
		cvLog( image_Re, image_Re ); // log(1 + Mag)

		cvMinMaxLoc(image_Re, &m, &M, NULL, NULL, NULL);
		cvScale(image_Re, image_Re, 1.0/(M-m), 1.0*(-m)/(M-m));
		minVal = 0; maxVal = 0;
		cvMinMaxLoc(image_Re, &minVal, &maxVal, &minLoc, &maxLoc, 0);
		cvCvtScaleAbs(image_Re,image_Re1,255.0*(maxVal-minVal),0);
		//cvNamedWindow("inverse",0);
		//cvShowImage("inverse",image_Re1);
		//cvWaitKey(-1);

		cvReleaseImage(&image_Re);
		cvReleaseImage(&image_Re1);
		cvReleaseImage(&image_Im);
		cvReleaseImage(&im);
		cvReleaseImage(&im1);
		cvReleaseImage(&realInput);
		cvReleaseImage(&imaginaryInput);
		cvReleaseImage(&complexInput);
		cvReleaseMat(&dft_A);
	}
	else if(atoi(argv[1])==2)
	{
		IplImage * magImage = cvLoadImage(argv[2],0);
		IplImage * phaseImage = cvLoadImage(argv[3],0);
		IplImage * realInput = cvCreateImage( cvGetSize(magImage), IPL_DEPTH_64F, 1);
		IplImage * imaginaryInput = cvCreateImage( cvGetSize(magImage), IPL_DEPTH_64F, 1);
		IplImage * complexInput = cvCreateImage( cvGetSize(magImage), IPL_DEPTH_64F, 2);
		IplImage * realInput2 = cvCreateImage( cvGetSize(phaseImage), IPL_DEPTH_64F, 1);
		IplImage * imaginaryInput2 = cvCreateImage( cvGetSize(phaseImage), IPL_DEPTH_64F, 1);
		IplImage * complexInput2 = cvCreateImage( cvGetSize(phaseImage), IPL_DEPTH_64F, 2);

		cvScale(magImage, realInput, 1.0, 0.0);
		cvZero(imaginaryInput);
		cvMerge(realInput, imaginaryInput, NULL, NULL, complexInput);
		int dft_M = cvGetOptimalDFTSize( (magImage->height > phaseImage->height?magImage->height:phaseImage->height) - 1 );
		int dft_N = cvGetOptimalDFTSize( (magImage->width > phaseImage->width?magImage->width:phaseImage->width) - 1 );
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

		cvScale(phaseImage, realInput2, 1.0, 0.0);
		cvZero(imaginaryInput2);
		cvMerge(realInput2, imaginaryInput2, NULL, NULL, complexInput2);
		//int dft_M2 = cvGetOptimalDFTSize( phaseImage->height - 1 );
		//int dft_N2 = cvGetOptimalDFTSize( phaseImage->width - 1 );
		CvMat* dft_A2, tmp2;
		dft_A2 = cvCreateMat( dft_M, dft_N, CV_64FC2 );
		cvGetSubRect( dft_A2, &tmp2, cvRect(0,0, phaseImage->width, phaseImage->height));
		cvCopy( complexInput2, &tmp2, NULL );
		if( dft_A2->cols > phaseImage->width )
		{
			cvGetSubRect( dft_A2, &tmp2, cvRect(phaseImage->width,0, dft_A2->cols -
						phaseImage->width, phaseImage->height));
			cvZero( &tmp2 );
		}
		cvDFT( dft_A2, dft_A2, CV_DXT_FORWARD, complexInput2->height );
		cvSplit( dft_A2, re, im, 0, 0 );


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
		cvSaveImage(argv[4],Output);
		//cvNamedWindow("impofphase",0);
		//cvShowImage("impofphase",Output);
		//cvWaitKey(-1);


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
	}
	else if(atoi(argv[1])==3)
	{
		IplImage * im = cvLoadImage(argv[2],0);
		IplImage * realInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
		IplImage * imaginaryInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
		IplImage * complexInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 2);
		CvMat* dft_A, tmp;
		int theta = atoi(argv[4]);
		int rad = atoi(argv[5]);
		int deltheta = atoi(argv[6]);
		int delrad = atoi(argv[7]);
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
						cvSet2D(re,i,j,s);
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
							cvSet2D(re,i,j,s);
						}
						else if ((val>=val1 && val<=val2) || (val>=val3 && val<=val4))
						{
							s.val[0]=0.0;
							cvSet2D(imag,i,j,s);
							cvSet2D(re,i,j,s);
						}
					}
				}
			}
		}

		//cvNamedWindow("imaginary",0);
		//cvShowImage("imaginary",imag);
		//cvNamedWindow("real",0);
		//cvShowImage("real",re);

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
		cvSaveImage(argv[3],Output);

		//cvNamedWindow("advfourier",0);
		//cvShowImage("advfourier",Output);

		//cvWaitKey(-1);

		cvReleaseImage(&im);
		cvReleaseImage(&realInput);
		cvReleaseImage(&imaginaryInput);
		cvReleaseImage(&complexInput);
		cvReleaseImage(&re);
		cvReleaseImage(&imag);
		cvReleaseMat(&dft_A);
		cvReleaseImage(&Output);
	}

	return 0;
}
