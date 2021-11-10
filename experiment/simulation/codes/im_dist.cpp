#include<iostream>
#include<math.h>
#include<algorithm>
#include<queue>
#include<vector>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;
int maxi(int a, int b)
{
	return a>b?a:b;
}

IplImage *imCityBlock2(IplImage *img,int x1,int y1)
{
	int c = cvGetSize(img).width;
	int r = cvGetSize(img).height;

	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);

	float scale;
	int maxd;
	maxd = maxi( maxi(x1+y1, x1+abs(y1-c) ), maxi(abs(x1-r)+y1, abs(x1-r)+abs(y1-c)) );
	scale=255.0/maxd;

	CvScalar s;
	for (int i=0;i<r;i++)
	{
		for (int j=0;j<c;j++)
		{
			s=cvGet2D(img,i,j);
			if (s.val[0]==0)
				s.val[0] = (abs(i-x1)+abs(j-y1)) * scale;
			cvSet2D(final_img,i,j,s);
		}
	}

	return final_img;
}

IplImage *imChessBoard2(IplImage *img,int x1,int y1)
{
	int c = cvGetSize(img).width;
	int r = cvGetSize(img).height;

	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);

	float scale;
	int maxd;
	maxd = maxi( maxi( maxi(x1,y1), maxi(x1,abs(y1-c)) ), maxi( maxi(abs(x1-r),y1), maxi(abs(x1-r),abs(y1-c))) );
	scale=255.0/maxd;

	CvScalar s;
	for (int i=0;i<r;i++)
	{
		for (int j=0;j<c;j++)
		{
			s=cvGet2D(img,i,j);
			if (s.val[0]==0)
				s.val[0] = maxi(abs(i-x1),abs(j-y1)) * scale;
			cvSet2D(final_img,i,j,s);
		}
	}

	return final_img;
}

IplImage* imEuclidean2(IplImage *img,int x1,int y1)
{
	int c = cvGetSize(img).width;
	int r = cvGetSize(img).height;

	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);

	float scale;
	int  maxd;
	maxd = maxi( maxi( sqrt( x1*x1 + y1*y1 ), sqrt( x1*x1 + (y1-c)*(y1-c) ) ), maxi( sqrt( (x1-r)*(x1-r) + y1*y1 ), sqrt( (x1-r)*(x1-r) + (y1-c)*(y1-c) ) ) );
	scale=255.0/maxd;

	CvScalar s;
	for (int i=0;i<r;i++)
	{
		for (int j=0;j<c;j++)
		{
			s=cvGet2D(img,i,j);
			if (s.val[0]==0)
				s.val[0] = sqrt( (i-x1)*(i-x1) + (j-y1)*(j-y1) ) * scale;
			cvSet2D(final_img,i,j,s);
		}
	}

	return final_img;
}

IplImage *imCityBlock(IplImage *img,int x1,int y1,int dist)
{
	int c = cvGetSize(img).width;
	int r = cvGetSize(img).height;

	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,3);
	cvMerge(img,img,img,NULL,final_img);

	int x2,y2,d;
	CvScalar s;

	s.val[0]=255;
	cvSet2D(final_img,x1,y1,s);
	s.val[0]=0;

	s.val[1]=255;
	for (x2=x1-dist; x2<=x1+dist; x2++)
	{
		d=dist-abs(x1-x2);

		y2=y1+d;
	//	cout<<x2<<","<<y2<<endl;
		if (x2<r && x2>=0 && y2<c && y2>=0)
			cvSet2D(final_img,x2,y2,s);

		y2=y1-d;
	//	cout<<x2<<","<<y2<<endl;
		if (x2<r && x2>=0 && y2<c && y2>=0)
			cvSet2D(final_img,x2,y2,s);
	}

	return final_img;
}

IplImage *imChessBoard(IplImage *img,int x1,int y1,int dist)
{
	int c = cvGetSize(img).width;
	int r = cvGetSize(img).height;

	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,3);
	cvMerge(img,img,img,NULL,final_img);

	int x2,y2;
	CvScalar s;

	s.val[0]=255;
	cvSet2D(final_img,x1,y1,s);
	s.val[0]=0;

	s.val[1]=255;
	for (y2=y1-dist; y2<=y1+dist; y2++)
	{
		x2=x1+dist;
		if (x2<r && x2>=0 && y2<c && y2>=0)
			cvSet2D(final_img,x2,y2,s);

		x2=x1-dist;
		if (x2<r && x2>=0 && y2<c && y2>=0)	
		cvSet2D(final_img,x2,y2,s);
	}
	for (x2=x1-dist; x2<=x1+dist;x2++)
	{
		y2=y1+dist;
		if (x2<r && x2>=0 && y2<c && y2>=0)
			cvSet2D(final_img,x2,y2,s);

		y2=y1-dist;
		if (x2<r && x2>=0 && y2<c && y2>=0)
			cvSet2D(final_img,x2,y2,s);
	}

	return final_img;
}

IplImage* imEuclidean(IplImage *img,int x1,int y1,int dist)
{
	int c = cvGetSize(img).width;
	int r = cvGetSize(img).height;

	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,3);
	cvMerge(img,img,img,NULL,final_img);

	int x2,y2,d;
	CvScalar s;

	s.val[0]=255;
	cvSet2D(final_img,x1,y1,s);
	s.val[0]=0;

	s.val[1]=255;
	for (x2=x1-dist; x2<=x1+dist; x2++)
	{
		d=(int)sqrt( (dist*dist) - ((x1-x2)*(x1-x2)) );
		y2=y1+d;
		if (x2<r && x2>=0 && y2<c && y2>=0)
			cvSet2D(final_img,x2,y2,s);

		y2=y1-d;
		if (x2<r && x2>=0 && y2<c && y2>=0)
		cvSet2D(final_img,x2,y2,s);
	}

	return final_img;
}


int main(int argc,char* argv[])
{
	int choice = atoi (argv[1]) ;
	
	if(choice==1) {
	int x1 = atoi(argv[2]);
	int y1 = atoi(argv[3]);
	int x2 = atoi(argv[4]);
	int y2 = atoi(argv[5]);
	if(atoi(argv[6])==1)
	{
		// city block

	    cout<<abs(x2-x1)+abs(y2-y1);
	}
			
	if(atoi(argv[6])==2)
	{
		// chessboard

		cout<<maxi(abs(x2-x1),abs(y2-y1));
	}
	
	if(atoi(argv[6])==3)
	{
		// eulcidean

		printf("%.2f",(float)sqrt((float)(x1-x2)*(float)(x1-x2)+(float)(y1-y2)*(float)(y1-y2))); 
	}
	}
	
	if(choice==2) {
	IplImage *source = cvLoadImage(argv[2],0);

	int c = cvGetSize(source).width;
	int r = cvGetSize(source).height;

	int x1 = atoi(argv[4]);
	int y1 = atoi(argv[5]);
	int dist=atoi(argv[6]);

	IplImage* out_img = cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,3);
	
	if(atoi(argv[7])==1)
	{
		// city block

		out_img = imCityBlock(source,x1,y1,dist);
	}
			
	if(atoi(argv[7])==2)
	{
		// chessboard

		out_img = imChessBoard(source,x1,y1,dist);
	}
	
	if(atoi(argv[7])==3)
	{
		// eulcidean

		out_img = imEuclidean(source,x1,y1,dist);
	}
	cvSaveImage(argv[3],out_img);
	cvReleaseImage( &source);
	cvReleaseImage( &out_img );
	}
	
	if(choice==3) {
	IplImage *source = cvLoadImage(argv[2],0);

	int c = cvGetSize(source).width;
	int r = cvGetSize(source).height;

	int x1 = atoi(argv[4]);
	int y1 = atoi(argv[5]);
	int distmetric=atoi(argv[6]);
	
	IplImage* out_img = cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
	
	if(distmetric==1)
	{
		// city block

		out_img = imCityBlock2(source,x1,y1);
	}

	if(distmetric==2)
	{
		// chessboard

		out_img = imChessBoard2(source,x1,y1);
	}

	if(distmetric==3)
	{
		// eulcidean

		out_img = imEuclidean2(source,x1,y1);
	}

	cvSaveImage(argv[3],out_img);
	cvReleaseImage( &source);
	cvReleaseImage( &out_img );
	}
	
	
	return 0;
}
