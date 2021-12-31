#include<iostream>
#include<stdio.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;

IplImage *imtranslate(IplImage *src,int tx,int ty)
{
	IplImage *tmp=cvCreateImage(cvSize(src->width,src->height),IPL_DEPTH_8U,1);
/*	if(tx==0 and ty==0)
	{
		tmp= cvCloneImage(src);
		return tmp;
	}
*/
	for(int i=0;i< src->height-1-ty;i++)
	{
		for(int j=0;j<src->width-1-tx;j++)
		{
				CvScalar s1;
				s1= cvGet2D(src,i,j);
				cvSet2D(tmp,i+ty,j+tx,s1);
		}

	}
	return tmp;

}

IplImage *imscale_rotate(IplImage *src,float angl,float scale,int flag)
{
	CvPoint2D32f srcTri[3], dstTri[3];
	CvMat* rot_mat = cvCreateMat(2,3,CV_32FC1);
	CvMat* warp_mat = cvCreateMat(2,3,CV_32FC1);
	IplImage *dst;
	dst = cvCloneImage( src );

	srcTri[0].x = 0;
	srcTri[0].y = 0;
	srcTri[1].x = src->width - 1;
	srcTri[1].y = 0;
	srcTri[2].x = 0;
	srcTri[2].y = src->height - 1;

	dstTri[0].x = 0;
	dstTri[0].y = 0;
	dstTri[1].x = src->width - 1;
	dstTri[1].y = 0;
	dstTri[2].x = 0;
	dstTri[2].y = src->height - 1;

	cvGetAffineTransform( srcTri, dstTri, warp_mat );
	cvWarpAffine( src, dst, warp_mat,flag);
	cvCopy ( dst, src );


	// Compute rotation matrix
	CvPoint2D32f center = cvPoint2D32f( src->width/2, src->height/2 );
	cv2DRotationMatrix( center, angl, scale, rot_mat );

	// Do the transformation
	cvWarpAffine( src, dst, rot_mat );
	return dst;
}

int main(int argc,char* argv[])
{
	// Set up variables
	 
	IplImage *src, *dst;

	// Load image
	src=cvLoadImage(argv[1]);



	char *img_nm,*img_typ;
	img_nm = strtok (argv[2],".");
	img_typ = strtok (NULL, ".");

	// get angle of rotation, translation(%) and scale value for the transformation

	float angl;
	int tx,ty;
	float scale;

	angl = atof(argv[3]);
	tx = atoi(argv[4]);
	ty = atoi(argv[5]);
	scale = atof(argv[6]);
	scale = pow(2,scale);
/*
	cvNamedWindow( name, 1 );
	cvShowImage( name,src);
	cvWaitKey(0);
	return 0;
*/
	int flag[3];
	flag[0] = CV_INTER_NN+CV_WARP_FILL_OUTLIERS;
	flag[1] = CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS;
	flag[2] = CV_INTER_CUBIC+CV_WARP_FILL_OUTLIERS;

	
	for(int i = 0;i<3;i++)
	{
		dst = cvCloneImage( src );
//		tmp=cvCreateImage(cvSize(src->width,src->height),IPL_DEPTH_8U,1);
		dst->origin = src->origin;
		cvZero( dst );
		if(tx==0 and ty==0)		
			dst = imscale_rotate(src,angl,scale,flag[i]);
		else
			dst = imtranslate(src,tx,ty);
//		src = cvCloneImage(tmp);

		char nm[50];
		int n = sprintf(nm,"%s%d.%s",img_nm,i+1,img_typ);
		cvSaveImage(nm,dst);
	}

	cvReleaseImage( &src );
//	cvReleaseImage( &tmp );
	cvReleaseImage( &dst );

	return 0;
}

