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
			IplImage *p1=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	IplImage *p2=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	IplImage *p3=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	IplImage* out_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,3);
	
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
	cvMerge(p1,p2,p3,NULL,out_img);
	
	cvSaveImage("outimage",out_img);
	cvReleaseImage(&out_img);
	cvReleaseImage(&source);
}