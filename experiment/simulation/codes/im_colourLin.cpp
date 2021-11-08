#include<iostream>
#include<stdio.h>
#include <string.h>
#include <cmath>
#include<algorithm>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;

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

// argv[7] will tells on which plane the proc. needs to be done --> input should be 1 or 2 or 3
// and argv[8] tells which color model to select --> input should be "1" for RGB, "2" for HSV, and "3" for CMY

int main(int argc,char *argv[])
{
	int p[3];
	p[0] = CV_IMWRITE_JPEG_QUALITY;
	p[1] = 90;
	p[2] = 0;

	char *img_nm,*img_typ;
	IplImage* source = cvLoadImage(argv[1],1);
	IplImage* dst;
	IplImage* dst1;

	char *pch,*rem,*f_img;
	pch = strtok (argv[1],"/");
	while (pch != NULL)
	{
		f_img = pch;
		pch = strtok (NULL, "/");

	}

	img_nm = strtok (f_img,".");
	img_typ = strtok (NULL, ".");
	cout<<"hello worldhhhh"<<endl;
	int c=cvGetSize(source).width;
	int r=cvGetSize(source).height;
	cout<<"hello world"<<endl;	
	IplImage* src=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,3);
	IplImage* p1=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	IplImage* p2=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	IplImage* p3=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);

        if (atoi(argv[6])==1)
        {
                // processing the image in RGB
                CvScalar s,t;
                t.val[0]=0;
                t.val[1]=0;
                t.val[2]=0;
                for (int i=0;i<r;i++)
                {
                        for (int j=0;j<c;j++)
                        {
                                s=cvGet2D(source,i,j);
                                t.val[0]=s.val[0];
                                cvSet2D(p1,i,j,t);
                                t.val[0]=s.val[1];
                                cvSet2D(p2,i,j,t);
                                t.val[0]=s.val[2];
                                cvSet2D(p3,i,j,t);
                        }
                }
                cvMerge(p1,p2,p3,NULL,src);
        }
        else if (atoi(argv[6])==2)
        {
                // processing the image in HSV
                cvCvtColor(source,src,CV_RGB2HSV);
                cvSplit(src,p1,p2,p3,0);
        }
        else if (atoi(argv[6])==3)
        {
                // processing the image in CMY
                CvScalar s,t;
                t.val[0]=0;
                t.val[1]=0;
                t.val[2]=0;
                for (int i=0;i<r;i++)
                {
                        for (int j=0;j<c;j++)
                        {
                                s=cvGet2D(source,i,j);
                                t.val[0]=(s.val[2]+s.val[1])/2;
                                cvSet2D(p1,i,j,t);
                                t.val[0]=(s.val[0]+s.val[2])/2;
                                cvSet2D(p2,i,j,t);
                                t.val[0]=(s.val[0]+s.val[1])/2;
                                cvSet2D(p3,i,j,t);
                        }
                }
                cvMerge(p1,p2,p3,NULL,src);
        }


	float a,b;
	a = atof(argv[4]);
	b = atof(argv[5]);
	dst=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	dst1=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	dst1=Contrast_Stretch(src,a,b,atoi(argv[3]));


	cvSplit(dst1,p1,p2,p3,0);

        if (atoi(argv[6])==1)
        {
                // processing the image in RGB
                CvScalar s,t;
                t.val[0]=0;
                t.val[1]=0;
                t.val[2]=0;
                for (int i=0;i<r;i++)
                {
                        for (int j=0;j<c;j++)
                        {
                                s=cvGet2D(dst1,i,j);
                                t.val[0]=s.val[0];
                                cvSet2D(p1,i,j,t);
                                t.val[0]=s.val[1];
                                cvSet2D(p2,i,j,t);
                                t.val[0]=s.val[2];
                                cvSet2D(p3,i,j,t);
                        }
                }
                cvMerge(p1,p2,p3,NULL,dst);
        }
        else if (atoi(argv[6])==2)
        {
                // processing the image in HSV
                cvCvtColor(dst1,dst,CV_HSV2RGB);
                cvSplit(dst,p1,p2,p3,0);
        }
        else if (atoi(argv[6])==3)
        {
                // processing the image in CMY
                CvScalar s,t;
                t.val[0]=0;
                t.val[1]=0;
                t.val[2]=0;
                for (int i=0;i<r;i++)
                {
                        for (int j=0;j<c;j++)
                        {
                                s=cvGet2D(dst1,i,j);
                                t.val[0]=(s.val[2]+s.val[1])/2;
                                cvSet2D(p1,i,j,t);
                                t.val[0]=(s.val[0]+s.val[2])/2;
                                cvSet2D(p2,i,j,t);
                                t.val[0]=(s.val[0]+s.val[1])/2;
                                cvSet2D(p3,i,j,t);
                        }
                }
                cvMerge(p1,p2,p3,NULL,dst);
        }

	cvSaveImage(argv[2],dst);

	cvReleaseImage( &p1 );
	cvReleaseImage( &p2 );
	cvReleaseImage( &p3 );
	cvReleaseImage( &src );
	cvReleaseImage( &dst);
	cvReleaseImage( &dst1);

	return 0;
}
