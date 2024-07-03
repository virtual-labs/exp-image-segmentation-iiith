#include<iostream>
#include<stdio.h>
#include <string.h>
#include<opencv/cv.h>
#include<cmath>
#include<opencv/highgui.h>

using namespace std;

IplImage* imADD(IplImage *img1,IplImage *img2,int map_fn)
{
	int c = cvGetSize(img1).width;
	int r = cvGetSize(img1).height;
	
	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	if(map_fn==1)
	{
		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				int grval = (int)(s1.val[0] + s2.val[0]);
				if(grval > 255)
					grval=255;
				s.val[0] = grval;
				cvSet2D(final_img,i,j,s);
			}
		}

	}
	else if(map_fn==2)
	{
		int min_val=100000;
		int max_val=-300;

		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				int grval = (int)(s1.val[0] + s2.val[0]);
				if(grval > max_val)
					max_val = grval;
				if(grval < min_val)
					min_val = grval;
			}
		}
		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				int grval = (int)(s1.val[0] + s2.val[0]);
				if(max_val==0 and min_val==0)
					grval = 0;
				else
					grval = (int)(((grval - min_val)*255)/(max_val - min_val));
				s.val[0] = grval;
				cvSet2D(final_img,i,j,s);
			}
		}

	}
	return final_img;
}

IplImage* imSUB(IplImage *img1,IplImage *img2,int map_fn)
{
	int c = cvGetSize(img1).width;
	int r = cvGetSize(img1).height;
	
	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	if(map_fn==1)
	{
		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				int grval = (int)(s1.val[0] - s2.val[0]);
				if(grval < 0)
					grval=0;
				s.val[0] = grval;
				cvSet2D(final_img,i,j,s);
			}
		}

	}
	else if(map_fn==2)
	{
		int min_val=100000;
		int max_val=-300;

		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				int grval = (int)(s1.val[0] - s2.val[0]);
				if(grval > max_val)
					max_val = grval;
				if(grval < min_val)
					min_val = grval;
			}
		}
		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				int grval = (int)(s1.val[0] - s2.val[0]);
				if(max_val==0 and min_val==0)
					grval = 0;
				else
					grval = (int)((grval - min_val)*255/(max_val - min_val));
				s.val[0] = grval;
				cvSet2D(final_img,i,j,s);
			}
		}

	}
	return final_img;
}
IplImage* imDIFF(IplImage *img1,IplImage *img2,int map_fn)
{
	int c = cvGetSize(img1).width;
	int r = cvGetSize(img1).height;
	
	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	if(map_fn==1)
	{
		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				int grval = abs(s1.val[0] - s2.val[0]);
				s.val[0] = grval;
				cvSet2D(final_img,i,j,s);
			}
		}

	}
	else if(map_fn==2)
	{
		int min_val=100000;
		int max_val=-300;

		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				int grval = abs(s1.val[0] - s2.val[0]);
				if(grval > max_val)
					max_val = grval;
				if(grval < min_val)
					min_val = grval;
			}
		}
		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				int grval = abs(s1.val[0] - s2.val[0]);
				if(max_val==0 and min_val==0)
					grval = 0;
				else
					grval = (int)((grval - min_val)*255/(max_val - min_val));
				s.val[0] = grval;
				cvSet2D(final_img,i,j,s);
			}
		}

	}
	return final_img;
}

IplImage* imMULTIPLY(IplImage *img1,IplImage *img2,int map_fn)
{
	int c = cvGetSize(img1).width;
	int r = cvGetSize(img1).height;
	
	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	if(map_fn==1)
	{
		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				int grval = (int)(s1.val[0] * s2.val[0]);
				if(grval > 255)
					grval=255;
				s.val[0] = grval;
				cvSet2D(final_img,i,j,s);
			}
		}

	}
	else if(map_fn==2)
	{
		int min_val=100000;
		int max_val=-300;

		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				int grval = (int)(s1.val[0] * s2.val[0]);
				if(grval > max_val)
					max_val = grval;
				if(grval < min_val)
					min_val = grval;
			}
		}
		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				int grval = (int)(s1.val[0] * s2.val[0]);
				if(max_val==0 and min_val==0)
					grval = 0;
				else
					grval = (int)((grval - min_val)*255/(max_val - min_val));
				s.val[0] = grval;
				cvSet2D(final_img,i,j,s);
			}
		}

	}
	return final_img;
}

IplImage* imDIVIDE(IplImage *img1,IplImage *img2,int map_fn)
{
	int c = cvGetSize(img1).width;
	int r = cvGetSize(img1).height;
	
	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	if(map_fn==1)
	{
		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				double grval;
				if(s2.val[0]!=0)
					grval = s1.val[0]/s2.val[0];
				else
					grval = s1.val[0]/.5;
				if(grval > 255.0)
					grval=255;
				s.val[0] = (int)(grval);
				cvSet2D(final_img,i,j,s);
			}
		}

	}
	else if(map_fn==2)
	{
		double min_val=1000000;
	 	double  max_val=-300;

		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				double grval;
				if(s2.val[0]!=0)
					grval = s1.val[0]/s2.val[0];
				else
					grval = s1.val[0]/.5;
				if(grval > max_val)
					max_val = grval;
				if(grval < min_val)
					min_val = grval;
			}
		}
//		cout<<max_val<<" "<<min_val<<endl;
		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				double grval;
				if(s2.val[0]!=0)
					grval = s1.val[0]/s2.val[0];
				else
					grval = s1.val[0]/0.5;
//				cout<<grval<<" ";
				if(max_val==0 and min_val==0)
					grval = 0;
				else
				{
					grval = ((grval - min_val)*255.0)/(max_val - min_val);
//					cout<<grval<<" ";
				}
				s.val[0] = (int)(grval);
//				cout<<s.val[0]<<endl;
				cvSet2D(final_img,i,j,s);
			}
		}

	}
	return final_img;
}

int main(int argc,char *argv[])
{


	char *img_nm,*img_typ;
	IplImage* img1 = cvLoadImage(argv[1],0);
	

	char *pch,*rem,*f_img;
	pch = strtok (argv[1],"/");
	while (pch != NULL)
	{
		f_img = pch;
		pch = strtok (NULL, "/");

	}

	img_nm = strtok (f_img,".");
	img_typ = strtok (NULL, ".");

	int c1 = cvGetSize(img1).width;
	int r1 = cvGetSize(img1).height;


	IplImage* img2 = cvLoadImage(argv[2],0);

	char *imgnm2,*imgtyp2;
	pch = strtok (argv[2],"/");
	while (pch != NULL)
	{
		f_img = pch;
		pch = strtok (NULL, "/");

	}

	imgnm2 = strtok (f_img,".");
	imgtyp2 = strtok (NULL, ".");
	

	int map_fn = atoi(argv[5]);
	if(atoi(argv[4])==1)
	{
		IplImage* add_img=cvCreateImage(cvSize(c1,r1),IPL_DEPTH_8U,1);
		add_img=imADD(img1,img2,map_fn);
		cvSaveImage(argv[3],add_img);
		cvReleaseImage( &add_img );
	}
	if(atoi(argv[4])==2)
	{
		IplImage* sub_img=cvCreateImage(cvSize(c1,r1),IPL_DEPTH_8U,1);
		sub_img=imSUB(img1,img2,map_fn);
		cvSaveImage(argv[3],sub_img);
		cvReleaseImage( &sub_img );
	}

	if(atoi(argv[4])==3)
	{
		IplImage* diff_img=cvCreateImage(cvSize(c1,r1),IPL_DEPTH_8U,1);
		diff_img=imDIFF(img1,img2,map_fn);
		cvSaveImage(argv[3],diff_img);
		cvReleaseImage( &diff_img );
	}

	if(atoi(argv[4])==4)
	{
		IplImage* mul_img=cvCreateImage(cvSize(c1,r1),IPL_DEPTH_8U,1);
		mul_img=imMULTIPLY(img1,img2,map_fn);
		cvSaveImage(argv[3],mul_img);
		cvReleaseImage( &mul_img );
	}
	if(atoi(argv[4])==5)
	{
		IplImage* div_img=cvCreateImage(cvSize(c1,r1),IPL_DEPTH_8U,1);
		div_img=imDIVIDE(img1,img2,map_fn);
		cvSaveImage(argv[3],div_img);
		cvReleaseImage( &div_img );
	}
	cvReleaseImage( &img1 );
	cvReleaseImage( &img2 );
	return 0;
}
