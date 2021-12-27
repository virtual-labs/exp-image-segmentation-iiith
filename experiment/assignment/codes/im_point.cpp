#include<stdio.h>
#include <string.h>
#include <math.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;

IplImage* Log_Image(IplImage *src,int cn)
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
				int grval = s1.val[0];
				grval = cn*log10(1 + grval);
				if(grval > 255)
					grval= 255;
				if(grval < 0)
					grval = 0;
				s.val[0] = grval;
				cvSet2D(final_img,i,j,s);
		}
	}
	return final_img;
}

IplImage* Contrast_Stretch(IplImage *src,float a,float b)
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
				int grval = (int)(tan(a*3.14/180)*s1.val[0] + b);
				if(grval > 255)
					grval = 255;
				else if(grval < 0)
					grval = 0;
				s.val[0] = grval;
				cvSet2D(final_img,i,j,s);

		}
	}
	return final_img;
}

IplImage* Clipped_Image(IplImage *src,float a,float b,float beta)
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
				int grval = (int)(s1.val[0]);
				if(grval < (int)(a))
					grval = 0;
				else if(grval > (int)(b))
					grval = 255;
				else 
					grval = (int)(beta * grval);
				if(grval > 255)
					grval = 255;
				s.val[0] = grval;
				cvSet2D(final_img,i,j,s);
		}
	}
	return final_img;
}

IplImage* Windowed_Image(IplImage *src,float a,float b,float beta)
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
				int grval = (int)(s1.val[0]);
				if((grval < (int)(a)) or (grval > (int)(b)))
					grval = 0;
				else 
					grval = (int)(beta * grval);
				if(grval > 255)
					grval = 255;
				s.val[0] = grval;
				cvSet2D(final_img,i,j,s);
		}
	}
	return final_img;
}

int main(int argc,char *argv[])
{
	int p[3];
	p[0] = CV_IMWRITE_JPEG_QUALITY;
	p[1] = 90;
	p[2] = 0;

	char *img_nm,*img_typ;
	IplImage* src = cvLoadImage(argv[1],0);
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


	if(atoi(argv[3])==1)
	{
//		cout<<"2\n";
		float a,b;
		a = atof(argv[4]);
		b = atof(argv[5]);
		dst=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
		dst=Contrast_Stretch(src,a,b);
	}
	if(atoi(argv[3])==2)
	{
//		cout<<"3\n";
		int c;
		c = atof(argv[4]);
		dst=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
		dst=Log_Image(src,c);
	}

	if(atoi(argv[3])==3)
	{
		float a,b,beta;
		a = atof(argv[4]);
		b = atof(argv[5]);
		beta = atof(argv[6]);
		dst=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
		dst=Clipped_Image(src,a,b,beta);
	}

	if(atoi(argv[3])==4)
	{
		float a,b,beta;
		a = atof(argv[4]);
		b = atof(argv[5]);
		float slope = atof(argv[6]);
		dst=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
		dst=Windowed_Image(src,a,b,slope);
	}



	cvSaveImage(argv[2],dst);
	cvReleaseImage( &src );
	cvReleaseImage( &dst);
	return 0;
}
