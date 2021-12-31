#include<iostream>
#include<queue>
#include<stdio.h>
#include <string.h>
#include<opencv/cv.h>
#include<cmath>
#include<opencv/highgui.h>

using namespace std;

IplImage* imPATH(IplImage *img,int xs,int ys,int xd,int yd)
{
	int c = cvGetSize(img).width;
	int r = cvGetSize(img).height;
	
	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);

	int dst[r][c];
	for(int i=0;i<r;i++)
		for(int j=0;j<c;j++)
			dst[i][j]=100000;
	int srcx[r][c];
	int srcy[r][c];
	queue<int> X;
	queue<int> Y;
	srcx[xs][ys]=xs;
	srcy[xs][ys]=ys;
	X.push(xs);
	Y.push(ys);
	dst[xs][ys]=0;
	int pth_found=0;

	int xp,yp;
	while(!X.empty())
	{
		xp = X.front();
		yp = Y.front();
//		cout<<xp<<" "<<yp<<endl;
		X.pop();
		Y.pop();
		if(xp==xd and yp==yd)
		{
			pth_found=1;
			cout<<"path found\n";
			break;
		}
		CvScalar s1,s2,s3,s4;
		s1.val[0]=-1;
		s2.val[0]=-1;
		s3.val[0]=-1;
		s4.val[0]=-1;
		if(xp-1 >=0)
			s1 = cvGet2D(img,xp-1,yp);
		if(yp+1 < c)
			s2 = cvGet2D(img,xp,yp+1);
		if(xp+1 < r)
			s3 = cvGet2D(img,xp+1,yp);
		if(yp-1 >=0)
			s4 = cvGet2D(img,xp,yp-1);
		if(s1.val[0] > 0)
		{
			if(dst[xp-1][yp] > dst[xp][yp] + 1)
			{
				dst[xp-1][yp]=dst[xp][yp]+1;
				srcx[xp-1][yp]=xp;
				srcy[xp-1][yp]=yp;
				X.push(xp-1);
				Y.push(yp);
			}
		}
		if(s2.val[0] > 0)
		{
			if(dst[xp][yp+1] > dst[xp][yp] + 1)
			{
				dst[xp][yp+1]=dst[xp][yp]+1;
				srcx[xp][yp+1]=xp;
				srcy[xp][yp+1]=yp;
				X.push(xp);
				Y.push(yp+1);
			}
		}
		if(s3.val[0] > 0)
		{
			if(dst[xp+1][yp] > dst[xp][yp] + 1)
			{
				dst[xp+1][yp]=dst[xp][yp]+1;
				srcx[xp+1][yp]=xp;
				srcy[xp+1][yp]=yp;
				X.push(xp+1);
				Y.push(yp);
			}
		}
		if(s4.val[0] > 0)
		{
			if(dst[xp][yp-1] > dst[xp][yp] + 1)
			{
				dst[xp][yp-1]=dst[xp][yp]+1;
				srcx[xp][yp-1]=xp;
				srcy[xp][yp-1]=yp;
				X.push(xp);
				Y.push(yp-1);
			}
		}
	}

	if(!pth_found)
	{
		cout<<"path not found\n";
		xd = xp;
		yd = yp;
		cout<<xd<<" "<<yd<<endl;
	}

	while(1)
	{
		CvScalar s;
		s.val[0] = 255;
		cvSet2D(final_img,xd,yd,s);
		if(xd==xs and yd == ys)
			break;
		xd = srcx[xd][yd];
		yd = srcy[xd][yd];
	}
	return final_img;
}

int main(int argc,char *argv[])
{
	char *img_nm,*img_typ;
	IplImage* img1 = cvLoadImage(argv[1],0);
	

/*	char *pch,*rem,*f_img;
	pch = strtok (argv[1],"/");
	while (pch != NULL)
	{
		f_img = pch;
		pch = strtok (NULL, "/");

	}

	img_nm = strtok (f_img,".");
	img_typ = strtok (NULL, ".");
*/

	int c1 = cvGetSize(img1).width;
	int r1 = cvGetSize(img1).height;

	int xs = atoi(argv[3]);
	int ys = atoi(argv[4]);
	int xd = atoi(argv[5]);
	int yd = atoi(argv[6]);
	CvScalar ss,sd;
	ss = cvGet2D(img1,xs,ys);
	cout<<ss.val[0]<<endl;
	sd = cvGet2D(img1,xd,yd);
	cout<<sd.val[0]<<endl;
	if(ss.val[0]>0 and sd.val[0]>0)
	{
		//	cout<<"here :S \n";
		IplImage* path_img=cvCreateImage(cvSize(c1,r1),IPL_DEPTH_8U,1);
		path_img=imPATH(img1,xs,ys,xd,yd);
		cvSaveImage(argv[2],path_img);
		cvReleaseImage( &path_img );
	}
	cvReleaseImage( &img1 );
	return 0;
}
