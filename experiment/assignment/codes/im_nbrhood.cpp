#include<iostream>
#include<stdio.h>
#include <string.h>
#include<opencv/cv.h>
#include<cmath>
#include<vector>
#include<queue>
#include<algorithm>
#include<opencv/highgui.h>

using namespace std;

float sum=0;
int parm; 

IplImage* linear(IplImage *img, float arr[][7])
{
  int c = cvGetSize(img).width;
  int r = cvGetSize(img).height;

  IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,img->nChannels);
  for(int xp=0;xp<r;xp++)
    {
      for(int yp=0;yp<c;yp++)
	{
	  CvScalar s,s2;
	  s = cvGet2D(img,xp,yp);
	  double val=0;
	  for(int i=0;i<parm;i++) {
	    for(int j=0;j<parm;j++) {
	      if(xp-(parm/2)+i>=0 && xp-(parm/2)+i<r && yp-(parm/2)+j>=0 && yp-(parm/2)+j<c) {
		s2 = cvGet2D(img,xp-(parm/2)+i,yp-(parm/2)+j);
		  val+=arr[i][j]*(float)s2.val[0];
		}
	    }
	  }
	  if(sum!=0)
	  s.val[0]=(int)(val/sum);
	  else
	    s.val[0]=(int)val;
	 	 


	  cvSet2D(final_img,xp,yp,s);
	}
    }
  return final_img;
}

IplImage* medn_filt(IplImage *img)
{
  int c = cvGetSize(img).width;
  int r = cvGetSize(img).height;

  IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,img->nChannels);
  for(int xp=0;xp<r;xp++)
    {
      for(int yp=0;yp<c;yp++)
	{
	  
	  vector<int> grvals;

	  CvScalar s,s2;
	  s = cvGet2D(img,xp,yp);
	  for(int i=0;i<parm;i++) {
	    for(int j=0;j<parm;j++) {
	      if(xp-(parm/2)+i>=0 && xp-(parm/2)+i<r && yp-(parm/2)+j>=0 && yp-(parm/2)+j<c) {
		s2 = cvGet2D(img,xp-(parm/2)+i,yp-(parm/2)+j);
		grvals.push_back((int)s2.val[0]);
	      }
	    }
	  }		

	  sort(grvals.begin(),grvals.end());
	  if(grvals.size()%2==0)
	    s.val[0] = (int)(0.5*(grvals[grvals.size()/2-1] + grvals[grvals.size()/2]) );
	  else
	    s.val[0] = (int)(grvals[grvals.size()/2-1]);

	  cvSet2D(final_img,xp,yp,s);
	}
    }
  return final_img;
}

void UnsharpMasking(IplImage *src, char* out_img)
{
	IplImage *UnsharpMaskingImage = cvCreateImage(cvGetSize(src), src->depth, src-> nChannels);
	IplImage *temp1 = cvCreateImage(cvGetSize(src), src->depth, src-> nChannels);
	IplImage *temp2 = cvCreateImage(cvGetSize(src), src->depth, src-> nChannels);

	cvSmooth( src, temp1, CV_GAUSSIAN, 27, 27);
	cvSub( src,temp1, temp2, NULL );
	
	int c = cvGetSize(src).width;
        int r = cvGetSize(src).height;

	CvScalar s1,s2,s3;
	for (int i=0;i<r;i++)
	{
		for (int j=0;j<c;j++)
		{
			//s1=cvGet2D(src,i,j);
			//s2=cvGet2D(temp1,i,j);
			//s3.val[0]=s1.val[0]-s2.val[0];
			s1=cvGet2D(temp2,i,j);
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

int main(int argc,char *argv[])
{
  char *img_nm,*img_typ;
  IplImage* img1 = cvLoadImage(argv[1],0);

  float arr[7][7];
  int c = cvGetSize(img1).width;
  int r = cvGetSize(img1).height;

  parm = atoi(argv[3]);
  
  if(parm==6) {
        UnsharpMasking(img1, argv[2]);
  } else {
  IplImage* nbrd_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,img1->nChannels);
  if(parm==2) {
    parm=atoi(argv[4]);
    nbrd_img=medn_filt(img1);
  }
  else {
    for(int i=0;i<parm;i++) {
      for(int j=0;j<parm;j++) {
	arr[i][j]=atof(argv[i*parm+j+4]);
	sum+= arr[i][j];
	  }
    }
    nbrd_img=linear(img1,arr);
  }


  cvSaveImage(argv[2],nbrd_img);
  cvReleaseImage( &nbrd_img );
  }
  cvReleaseImage( &img1 );
  return 0;
}
