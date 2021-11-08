#include<iostream>
#include<algorithm>
#include<queue>
#include<vector>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;

void UnsharpMasking(IplImage *src, char* out_img)
{
	IplImage *UnsharpMaskingImage = cvCreateImage(cvGetSize(src), src->depth, src-> nChannels);
	IplImage *temp1 = cvCreateImage(cvGetSize(src), src->depth, src-> nChannels);
	IplImage *temp2 = cvCreateImage(cvGetSize(src), src->depth, src-> nChannels);

	cvSmooth( src, temp1, CV_GAUSSIAN, 27, 27);
	cvSub( src,temp1, temp2, NULL );
	
	int c = cvGetSize(src).width;
        int r = cvGetSize(src).height;

	CvScalar s1;

	for (int i=0;i<r;i++)
	{
		for (int j=0;j<c;j++)
		{
			s1=cvGet2D(temp2,i,j);
cout<<s1.val[1]<<endl;
			s1.val[0]=s1.val[0]*1.5;
			s1.val[1]=s1.val[1]*1.5;
			s1.val[2]=s1.val[2]*1.5;
			cvSet2D(temp2,i,j,s1);
		}
	}

	cvAdd( temp2, src, UnsharpMaskingImage, NULL );

	cvSaveImage("smoothened.jpg",temp1);
	cvSaveImage("subtracted.jpg",temp1);
	cvReleaseImage (&temp1);
	cvReleaseImage (&temp2);
	cvSaveImage(out_img, UnsharpMaskingImage);
	cvReleaseImage(&UnsharpMaskingImage);
}
int main(int argc,char* argv[])
{
	IplImage *source = cvLoadImage(argv[1],0);
	cout<<source->nChannels<<endl;
        UnsharpMasking(source, argv[2]);
	return 0;
}
