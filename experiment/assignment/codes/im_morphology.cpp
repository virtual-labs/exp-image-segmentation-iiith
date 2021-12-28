#include<iostream>
#include<algorithm>
#include<queue>
#include<vector>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;

// argv[2] indicates Operation
// argv[3] indicates Shape
// argv[3] indicates Size

int main(int argc, char* argv[])
{
	IplImage* source = cvLoadImage(argv[1],0);

	int col = cvGetSize(source).width;
	int row = cvGetSize(source).height;

	IplImage* temp = cvCreateImage(cvSize(col,row),IPL_DEPTH_8U,1);
	IplImage* dest = cvCreateImage(cvSize(col,row),IPL_DEPTH_8U,1);

	int a=atoi(argv[3]);
	int b=atoi(argv[4]);
	int c=atoi(argv[5]);

	IplConvKernel* se;

	if (b==1)
	{
		if (c==1)
			se = cvCreateStructuringElementEx( 3, 3, 1, 1, CV_SHAPE_ELLIPSE, 0 );
		else if (c==2)
			se = cvCreateStructuringElementEx( 5, 5, 2, 2, CV_SHAPE_ELLIPSE, 0 );
		else if (c==3)
			se = cvCreateStructuringElementEx( 7, 7, 2, 2, CV_SHAPE_ELLIPSE, 0 );
	}
	else if (b==2)
	{
		if (c==1)
			se = cvCreateStructuringElementEx( 3, 3, 1, 1, CV_SHAPE_RECT, 0 );
		else if (c==2)
			se = cvCreateStructuringElementEx( 5, 5, 2, 2, CV_SHAPE_RECT, 0 );
		else if (c==3)
			se = cvCreateStructuringElementEx( 7, 7, 2, 2, CV_SHAPE_RECT, 0 );
	}
	else if (b==3)
	{
		if (c==1)
			se = cvCreateStructuringElementEx( 3, 3, 1, 1, CV_SHAPE_ELLIPSE, 0 );
		else if (c==2)
			se = cvCreateStructuringElementEx( 5, 5, 2, 2, CV_SHAPE_ELLIPSE, 0 );
		else if (c==3)
			se = cvCreateStructuringElementEx( 7, 7, 2, 2, CV_SHAPE_ELLIPSE, 0 );
	}

	if (a==1)
	{
		cvErode(source,dest,se,1);
	}
	else if (a==2)
	{
		cvDilate(source,dest,se,1);
	}
	else if (a==3)
	{
		cvErode(source,temp,se,1);
		cvDilate(temp,dest,se,1);
	}
	else if (a==4)
	{
		cvDilate(source,temp,se,1);
		cvErode(temp,dest,se,1);
	}

	CvScalar s1,s2;
	for (int i=0;i<row;i++)
	{
		for (int j=0;j<col;j++)
		{
			s1=cvGet2D(source,i,j);
			s2=cvGet2D(dest,i,j);
			if (s1.val[0]!=s2.val[0])
				cout<<s1.val[0]<<" and "<<s2.val[0]<<endl;
		}
	}

	cvSaveImage(argv[2],dest);
}
