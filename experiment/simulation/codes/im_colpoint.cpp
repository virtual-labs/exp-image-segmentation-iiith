#include<iostream>
#include<stdio.h>
#include <string.h>
#include <cmath>
#include<algorithm>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;


IplImage* Log_Image(IplImage *src,int cn, int arg)
{
	int c = cvGetSize(src).width;
	int r = cvGetSize(src).height;
	
	IplImage* final_img=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvZero(final_img);
	for(int i=0;i<r;i++)
	{
		for(int j=0;j<c;j++)
		{
				CvScalar s1,s;
				s1 = cvGet2D(src,i,j);
				s.val[0]=s1.val[0];
				s.val[1]=s1.val[1];
				s.val[2]=s1.val[2];

				if (arg==1 || arg==3 || arg==5 || arg==7)
				{
					int grval1 = s1.val[0];
					grval1 = cn*log10(1 + grval1);
					if(grval1 > 255)
						grval1= 255;
					if(grval1 < 0)
						grval1 = 0;
					s.val[0] = grval1;
				}

				if (arg==2 || arg==3 || arg==6 || arg==7)
				{
					int grval2 = s1.val[1];
					grval2 = cn*log10(1 + grval2);
					if(grval2 > 255)
						grval2= 255;
					if(grval2 < 0)
						grval2 = 0;
					s.val[1] = grval2;
				}

				if (arg==4 || arg==5 || arg==6 || arg==7)
				{
					int grval3 = s1.val[2];
					grval3 = cn*log10(1 + grval3);
					if(grval3 > 255)
						grval3= 255;
					if(grval3 < 0)
						grval3 = 0;
					s.val[2] = grval3;
				}

				cvSet2D(final_img,i,j,s);
		}
	}
	return final_img;
}

IplImage* Contrast_Stretch(IplImage *src,float a,float b, int arg)
{
	int c = cvGetSize(src).width;
	int r = cvGetSize(src).height;
	
	IplImage* final_img=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvZero(final_img);
	for(int i=0;i<r;i++)
	{
		for(int j=0;j<c;j++)
		{
				CvScalar s1,s;
				s1 = cvGet2D(src,i,j);
				s.val[0]=s1.val[0];
				s.val[1]=s1.val[1];
				s.val[2]=s1.val[2];

				if (arg==1 || arg==3 || arg==5 || arg==7)
				{
					int grval1 = (int)(tan(a*3.14/180)*s1.val[0] + b);
					if(grval1 > 255)
						grval1 = 255;
					else if(grval1 < 0)
						grval1 = 0;
					s.val[0] = grval1;
				}

				if (arg==2 || arg==3 || arg==6 || arg==7)
				{
					int grval2 = (int)(tan(a*3.14/180)*s1.val[1] + b);
					if(grval2 > 255)
						grval2 = 255;
					else if(grval2 < 0)
						grval2 = 0;
					s.val[1] = grval2;
				}

				if (arg==4 || arg==5 || arg==6 || arg==7)
				{
					int grval3 = (int)(tan(a*3.14/180)*s1.val[2] + b);
					if(grval3 > 255)
						grval3 = 255;
					else if(grval3 < 0)
						grval3 = 0;
					s.val[2] = grval3;
				}

				cvSet2D(final_img,i,j,s);

		}
	}
	return final_img;
}

IplImage* Clipped_Image(IplImage *src,float a,float b,float beta, int arg)
{
	int c = cvGetSize(src).width;
	int r = cvGetSize(src).height;
	
	IplImage* final_img=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvZero(final_img);
	for(int i=0;i<r;i++)
	{
		for(int j=0;j<c;j++)
		{
				CvScalar s1,s;
				s1 = cvGet2D(src,i,j);
				s.val[0]=s1.val[0];
				s.val[1]=s1.val[1];
				s.val[2]=s1.val[2];

				if (arg==1 || arg==4 || arg==6 || arg==7)
				{
					int grval1 = (int)(s1.val[0]);
					if(grval1 < (int)(a))
						grval1 = 0;
					else if(grval1 > (int)(b))
						grval1 = 255;
					else 
						grval1 = (int)(beta * grval1);
					if(grval1 > 255)
						grval1 = 255;
					s.val[0] = grval1;
				}

				if (arg==2 || arg==4 || arg==5 || arg==7)
				{
					int grval2 = (int)(s1.val[1]);
					if(grval2 < (int)(a))
						grval2 = 0;
					else if(grval2 > (int)(b))
						grval2 = 255;
					else 
						grval2 = (int)(beta * grval2);
					if(grval2 > 255)
						grval2 = 255;
					s.val[1] = grval2;
				}

				if (arg==3 || arg==5 || arg==6 || arg==7)
				{
					int grval3 = (int)(s1.val[2]);
					if(grval3 < (int)(a))
						grval3 = 0;
					else if(grval3 > (int)(b))
						grval3 = 255;
					else 
						grval3 = (int)(beta * grval3);
					if(grval3 > 255)
						grval3 = 255;
					s.val[2] = grval3;
				}

				cvSet2D(final_img,i,j,s);
		}
	}
	return final_img;
}

IplImage* Windowed_Image(IplImage *src,float a,float b,float beta, int arg)
{
	int c = cvGetSize(src).width;
	int r = cvGetSize(src).height;
	
	IplImage* final_img=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvZero(final_img);
	for(int i=0;i<r;i++)
	{
		for(int j=0;j<c;j++)
		{
				CvScalar s1,s;
				s1 = cvGet2D(src,i,j);
				s.val[0]=s1.val[0];
				s.val[1]=s1.val[1];
				s.val[2]=s1.val[2];

				if (arg==1 || arg==4 || arg==6 || arg==7)
				{
					int grval1 = (int)(s1.val[0]);
					if((grval1 < (int)(a)) or (grval1 > (int)(b)))
						grval1 = 0;
					else
						grval1 = (int)(beta * grval1);
					if(grval1 > 255)
						grval1 = 255;
					s.val[0] = grval1;
				}

				if (arg==2 || arg==4 || arg==5 || arg==7)
				{
					int grval2 = (int)(s1.val[1]);
					if((grval2 < (int)(a)) or (grval2 > (int)(b)))
						grval2 = 0;
					else 
						grval2 = (int)(beta * grval2);
					if(grval2 > 255)
						grval2 = 255;
					s.val[1] = grval2;
				}

				if (arg==3 || arg==5 || arg==6 || arg==7)
				{
					int grval3 = (int)(s1.val[2]);
					if((grval3 < (int)(a)) or (grval3 > (int)(b)))
						grval3 = 0;
					else 
						grval3 = (int)(beta * grval3);
					if(grval3 > 255)
						grval3 = 255;
					s.val[2] = grval3;
				}

				cvSet2D(final_img,i,j,s);
		}
	}
	return final_img;
}

 // argv[7] will tells on which plane the proc. needs to be done --> input should be 1 or 2 or 3
 // and argv[8] tells which color model to select --> input should be "1" for RGB, "2" for HSV, and "3" for CMY


int main(int argc,char *argv[])
{
	int p[3];
	p[0] = CV_IMWRITE_JPEG_QUALITY;
	p[1] = 90;
	p[2] = 0;

	char *img_nm,*img_typ;
	IplImage* src = cvLoadImage(argv[1],1);
	IplImage* dst;

	char *pch,*rem,*f_img;
	pch = strtok (argv[1],"/");
	while (pch != NULL)
	{
		f_img = pch;
		pch = strtok (NULL, "/");

	}

	img_nm = strtok (f_img,".");
	img_typ = strtok (NULL, ".");


//		cout<<"2\n";
		float a,b;
		a = atof(argv[3]);
		b = atof(argv[4]);
		dst=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
		dst=Contrast_Stretch(src,a,b,atoi(argv[5]));
		
	cvSaveImage(argv[2],dst);

	int c=cvGetSize(src).width;
	int r=cvGetSize(src).height;
	IplImage* p1=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	IplImage* p2=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	IplImage* p3=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	cvSplit(src,p1,p2,p3,0);
	cvSaveImage("p1.jpg",p1);
	cvSaveImage("p2.jpg",p2);
	cvSaveImage("p3.jpg",p3);
	cvSplit(dst,p1,p2,p3,0);
	cvSaveImage("newp1.jpg",p1);
	cvSaveImage("newp2.jpg",p2);
	cvSaveImage("newp3.jpg",p3);
	cvReleaseImage( &p1 );
	cvReleaseImage( &p2 );
	cvReleaseImage( &p3 );

	cvReleaseImage( &src );
	cvReleaseImage( &dst);
	return 0;
}
