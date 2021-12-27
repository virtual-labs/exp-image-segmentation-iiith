#include<iostream>
#include<stdio.h>
#include <string.h>
#include<algorithm>
#include<queue>
#include<vector>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;

int main(int argc,char *argv[])
{


	char *dir,*img_nm,*img_typ;
	IplImage* timg1=0;
	timg1 = cvLoadImage(argv[1],1);

	char *pch,*rem,*f_img;
	pch = strtok (argv[1],"/");
	while (pch != NULL)
	{
		f_img = pch;
		pch = strtok (NULL, "/");

	}

	img_nm = strtok (f_img,".");
	img_typ = strtok (NULL, ".");

	int c1 = 300;
	int r1 = 300;

	IplImage* img1=0;
	img1 = cvCreateImage(cvSize(c1,r1),timg1->depth,timg1->nChannels);
	cvResize(timg1,img1);
	cvReleaseImage( &timg1 );

	char rszimg1[50],rszimg2[50];
	int tn = sprintf(rszimg1,"uploads/resized_%s.%s",img_nm,img_typ);	
	cvSaveImage(rszimg1,img1);
	cvReleaseImage( &img1 );
	return 0;

}
