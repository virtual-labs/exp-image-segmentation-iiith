#include<iostream>
#include<stdio.h>
#include <string.h>
#include<algorithm>
#include<queue>
#include<vector>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;

IplImage* imOR(IplImage *img1,IplImage *img2)
{
	int c = cvGetSize(img1).width;
	int r = cvGetSize(img1).height;
	
	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	for(int i=0;i<r;i++)
	{
		for(int j=0;j<c;j++)
		{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				int grval=0;
//				if((s1.val[0] == 0) or (s2.val[0] == 0));
//					cout<<s1.val[0]<<" "<<s2.val[0]<<endl;
				if((s1.val[0] > 0) or (s2.val[0] > 0))
					grval = 255;
				s.val[0] = grval;
				cvSet2D(final_img,i,j,s);
		}
	}
	return final_img;
}

IplImage* imAND(IplImage *img1,IplImage *img2)
{
	int c = cvGetSize(img1).width;
	int r = cvGetSize(img1).height;
	
	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	for(int i=0;i<r;i++)
	{
		for(int j=0;j<c;j++)
		{
				CvScalar s1,s2,s;
				s1 = cvGet2D(img1,i,j);
				s2 = cvGet2D(img2,i,j);
				int grval=0;
				if((s1.val[0] > 0) and (s2.val[0] > 0))
					grval = 255;
	//			cout<<grval<<endl;
				s.val[0] = grval;
				cvSet2D(final_img,i,j,s);
		}
	}
	return final_img;
}

int main(int argc,char *argv[])
{


	char *img_nm,*img_typ;

	IplImage* timg1 = cvLoadImage(argv[1],0);


	img_nm = strtok (argv[1],".");
	img_typ = strtok (NULL, ".");

	int c1 = 300;
	int r1 = 300;

//	int c1 = cvGetSize(img1).width;
//	int r1 = cvGetSize(img1).height;
	cvThreshold(timg1,timg1,200,255,CV_THRESH_BINARY);


	IplImage* img2 = cvLoadImage(argv[2],0);

	char *imgnm2,*imgtyp2;
	imgnm2 = strtok (argv[2],".");
	imgtyp2 = strtok (NULL, ".");


//	int c2 = cvGetSize(img2).width;
//	int r2 = cvGetSize(img2).height;

	cvThreshold(img2,img2,150,255,CV_THRESH_BINARY);

//	if(r1!=r2 or c1!=c2)
//	{

		IplImage* img1=cvCreateImage(cvSize(c1,r1),IPL_DEPTH_8U,1);
		cvResize(timg1,img1,CV_INTER_LINEAR);
		cvReleaseImage( &timg1 );

		IplImage* timg2=cvCreateImage(cvSize(c1,r1),IPL_DEPTH_8U,1);
		cvResize(img2,timg2,CV_INTER_LINEAR);
		cvReleaseImage( &img2 );
		img2 = timg2;
		cvReleaseImage( &timg2 );
//	}




	IplImage* or_img=cvCreateImage(cvSize(c1,r1),IPL_DEPTH_8U,1);
	or_img=imOR(img1,img2);

	IplImage* and_img=cvCreateImage(cvSize(c1,r1),IPL_DEPTH_8U,1);
	and_img=imAND(img1,img2);

	char rszimg1[50],rszimg2[50];
	int tn = sprintf(rszimg1,"resized_%s.%s",img_nm,img_typ);	
	cvSaveImage(rszimg1,img1);
	tn = sprintf(rszimg2,"resized_%s.%s",imgnm2,imgtyp2);	
	cvSaveImage(rszimg2,img2);


	cvSaveImage("or.jpg",or_img);
	cvSaveImage("and.jpg",and_img);



/*	cvNamedWindow( "Example3-input1" );
	cvNamedWindow( "Example3-input2" );
	cvNamedWindow( "Example3-OR" );
	cvNamedWindow( "Example3-AND" );

	cvShowImage("Example3-input1",img1);
	cvShowImage("Example3-input2",img2);
	cvShowImage("Example3-OR",or_img);
	cvShowImage("Example3-AND",and_img);

	cvWaitKey(0);
 */

	cvReleaseImage( &img1 );
	cvReleaseImage( &img2 );
	cvReleaseImage( &or_img );
	cvReleaseImage( &and_img );
/*
	
	cvDestroyWindow( "Example3-input1" );
	cvDestroyWindow( "Example3-input2" );
	cvDestroyWindow( "Example3-OR" );
	cvDestroyWindow( "Example3-AND" );

*/
	return 0;
}
