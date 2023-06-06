#include<iostream>
#include<algorithm>
#include<queue>
#include<vector>
#include<opencv/cv.h>
#include<opencv/highgui.h>
using namespace std;
int main(int argc, char* argv[])
{
	IplImage *source=cvLoadImage(argv[1],1);
	int c=cvGetSize(source).width;
	int r=cvGetSize(source).height;
	IplImage* out_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,3);
	IplImage *p1=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	IplImage *p2=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	IplImage *p3=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	char *img_nm,*img_typ;
	img_nm = strtok (argv[2],".");
	img_typ = strtok (NULL, ".");
	if(atoi(argv[3])==1 || atoi(argv[3])==5)
	{
		// converting the RGB image to HSV		
		// cout<<"converting the RGB image to HSV"<<endl;
		cvCvtColor(source,out_img,CV_RGB2HSV);
		cvSplit(out_img,p1,p2,p3,0);
	}
	else if(atoi(argv[3])==2)
	{
		// converting the RGB image to CMY
		// cout<<"converting the RGB image to CMY"<<endl;
		CvScalar s,t;
		t.val[0]=0;
		t.val[1]=0;
		t.val[2]=0;
		for (int i=0;i<r;i++)
		{
			for (int j=0;j<c;j++)
			{
				s=cvGet2D(source,i,j);
				t.val[0]=(s.val[0]+s.val[1])/2;
				cvSet2D(p1,i,j,t);
				t.val[0]=(s.val[1]+s.val[2])/2;
				cvSet2D(p2,i,j,t);
				t.val[0]=(s.val[0]+s.val[2])/2;
				cvSet2D(p3,i,j,t);
			}		
		}
		cvMerge(p1,p2,p3,NULL,out_img);
	}
	else if(atoi(argv[3])==3)
	{
		// converting the RGB image to YCrCb
		// cout<<"converting the RGB image to YCrCb"<<endl;
		cvCvtColor(source,out_img,CV_RGB2YCrCb);
		cvSplit(out_img,p1,p2,p3,0);
	}
	else if(atoi(argv[3])==4)
	{
		cvSplit(source,p1,p2,p3,0);
	}
	char nam[30];
	int n;
	/*	if(atoi(argv[3])==5) {
		n = sprintf(nam,"%s4.%s",img_nm,img_typ);
		cvSaveImage(nam,p1);
		n = sprintf(nam,"%s5.%s",img_nm,img_typ);
		cvSaveImage(nam,p2);
		n = sprintf(nam,"%s6.%s",img_nm,img_typ);
		cvSaveImage(nam,p3);
	} else 
		{ */	
	n = sprintf(nam,"%s1.%s",img_nm,img_typ);
	cvSaveImage(nam,p1);
	n = sprintf(nam,"%s2.%s",img_nm,img_typ);
	cvSaveImage(nam,p2);
	n = sprintf(nam,"%s3.%s",img_nm,img_typ);
	cvSaveImage(nam,p3);  
	cvReleaseImage(&p1);
	cvReleaseImage(&p2);
	cvReleaseImage(&p3);
	cvReleaseImage(&source);
	cvReleaseImage(&out_img);
	return 0;
}
