#include<iostream>
#include<algorithm>
#include<queue>
#include<vector>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;

IplImage* resample(IplImage *out_img,int r,int c)
{
	IplImage* final_img=cvCreateImage(cvSize(r,c),IPL_DEPTH_8U,1);
	for(int i=0;i<c;i++)
	{
		for(int j=0;j<r;j++)
		{
			if(i%2==0)
			{
//				cout<<"i%2==0 I "<<i<<" "<<"J "<<j<<endl;
				CvScalar s1;
				s1= cvGet2D(out_img,i/2,j/2);
				cvSet2D(final_img,i,j,s1);
			}
			else
			{
//				cout<<"i%2==1 I "<<i<<" "<<"J "<<j<<endl;
				CvScalar s1;
				s1 = cvGet2D(final_img,i-1,j);
				cvSet2D(final_img,i,j,s1);
			}
		}
	}
	return final_img;
}


int main()
{
	int cx,cy;
//	cin>>cx>>cy;
	
	IplImage* img = cvLoadImage("green.jpg",0);
	int rf = cvGetSize(img).width;
	int cf = cvGetSize(img).height;
	int red;
	cin>>red;
	IplImage* out_img=cvCreateImage(cvSize(rf/pow(2,red),cf/pow(2,red)),IPL_DEPTH_8U,1);
	int r = cvGetSize(out_img).width;
	int c = cvGetSize(out_img).height;

	cout<<r<<" x "<<c<<endl;

	cvResize(img,out_img,CV_INTER_LINEAR);

	IplImage* final_img=cvCreateImage(cvSize(rf,cf),IPL_DEPTH_8U,1);
	IplImage* tmp_img;
	while(1)
	{
		if(red==0)
		{
			final_img = tmp_img;
			break;
		}
		r = cvGetSize(out_img).width;
		c = cvGetSize(out_img).height;

		tmp_img = resample(out_img,2*r,2*c);

		cout<<cvGetSize(tmp_img).width<<" x "<<cvGetSize(tmp_img).height<<endl;

		cvReleaseImage( &out_img );
		out_img = tmp_img;
		red = red - 1;
	}
	
	cvNamedWindow( "Example3-in" );
	cvNamedWindow( "Example3-1" );

	cvShowImage("Example3-1",img);
	cvShowImage("Example3-in",final_img);

	cvWaitKey(0);

	cvReleaseImage( &img );
	cvReleaseImage( &final_img );

	cvDestroyWindow( "Example3-in" );
	cvDestroyWindow( "Example3-1" );

	return 0;
}
