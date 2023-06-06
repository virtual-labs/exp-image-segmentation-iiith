#include<iostream>
#include<stdio.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;

IplImage *imtranslate(IplImage *src,int tx,int ty)
{
	IplImage *tmp=cvCreateImage(cvSize(src->width,src->height),IPL_DEPTH_8U,1);
	cvZero(tmp);
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
	CvPoint2D32f center = cvPoint2D32f( (src->width)/2, (src->height)/2 );
	cv2DRotationMatrix( center, angl, scale, rot_mat );

	// Do the transformation
	cvWarpAffine( src, dst, rot_mat );
	return dst;
}

int main(int argc,char* argv[])
{
	// Set up variables

	IplImage *src, *dst,*tmp,*tmp2;

	// Load image
	src=cvLoadImage(argv[1]);



	char *img_nm,*img_typ;
	img_nm = strtok (argv[2],".");
	img_typ = strtok (NULL, ".");

	//	cout<<argv[2]<<endl;
	// get angle of rotation, translation(%) and scale value for the transformation

	float angl;
	int tx,ty,sc_val;
	float scale;

	int sc_arg,ang_arg,interp;

	interp = atoi(argv[3]);
	sc_val = atoi(argv[4]);
	sc_arg = atoi(argv[5]);
	if(sc_arg==0)
		scale = pow(2,sc_val);
	else
		scale = pow(2,-sc_val);

	angl = atof(argv[6]);
	//	cout<<scale<<endl;

	tx = atoi(argv[7]);
	ty = atoi(argv[8]);

	int o1,o2,o3;
	o1 = atoi(argv[9]);
	o2 = atoi(argv[10]);
	o3 = atoi(argv[11]);
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

	int intrp[3];
	intrp[0] = CV_INTER_NN;
	intrp[1] = CV_INTER_LINEAR;
	intrp[2] = CV_INTER_CUBIC;


	/*		tmp2=cvCreateImage(cvSize((int)(scale)*src->width,(int)(scale)*src->height),IPL_DEPTH_8U,1);	
			cvZero( dst );*/
	int nt,opn_cnt = 1;
	char nam[50];

	if(o1==1)
	{
		if(scale==1)
			tmp = cvCloneImage(src);
		else
		{
			tmp=cvCreateImage(cvSize(cvRound(scale*src->width),cvRound(scale*src->height)),src->depth,src->nChannels);	
			cvZero( tmp );
			cvResize(src,tmp,intrp[interp-1]);
			nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
			cvSaveImage(nam,tmp);
			opn_cnt++;
		}

		if(o2==2 and o3==3)
		{
			if(angl==0)
				tmp2=cvCloneImage(tmp);
			else
			{		
				tmp2=cvCreateImage(cvSize(cvRound(scale*src->width),cvRound(scale*src->height)),src->depth,src->nChannels);	
				cvZero( tmp2 );
				tmp2 = imscale_rotate(tmp,angl,1,flag[interp-1]);
				nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
				cvSaveImage(nam,tmp2);
				opn_cnt++;

			}
			if(tx==0 and ty==0)
				dst=cvCloneImage(tmp2);
			else
			{
				dst=cvCreateImage(cvSize(cvRound(scale*src->width),cvRound(scale*src->height)),src->depth,src->nChannels);	
				cvZero( dst );
				dst = imtranslate(tmp2,tx,ty);
				nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
				cvSaveImage(nam,dst);
				opn_cnt++;
			}

		}
		else
		{
			if(tx==0 and ty==0)
				tmp2=cvCloneImage(tmp);
			else
			{
				tmp2=cvCreateImage(cvSize(cvRound(scale*src->width),cvRound(scale*src->height)),src->depth,src->nChannels);	
				cvZero( tmp2 );
				tmp2 = imtranslate(tmp,tx,ty);
				nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
				cvSaveImage(nam,tmp2);
				opn_cnt++;
			}
			if(angl==0)
				dst=cvCloneImage(tmp2);
			else
			{
				dst=cvCreateImage(cvSize(cvRound(scale*src->width),cvRound(scale*src->height)),src->depth,src->nChannels);	
				cvZero( dst );
				dst = imscale_rotate(tmp2,angl,1,flag[interp-1]);
				nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
				cvSaveImage(nam,dst);
				opn_cnt++;
			}

		}
	}

	else if(o2==1)
	{
		if(angl==0)
			tmp = cvCloneImage(src);
		else
		{
			tmp=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);	
			cvZero( tmp );
			tmp = imscale_rotate(src,angl,1,flag[interp-1]);
			nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
			cvSaveImage(nam,tmp);
			opn_cnt++;

		}
		if(o1==2 and o3==3)
		{
			if(scale==1)
				tmp2 = cvCloneImage(tmp);
			else
			{
				tmp2=cvCreateImage(cvSize(cvRound(scale*src->width),cvRound(scale*src->height)),src->depth,src->nChannels);	
				cvZero( tmp2 );
				cvResize(tmp,tmp2,intrp[interp-1]);
				nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
				cvSaveImage(nam,tmp2);
				opn_cnt++;

			}
			if(tx==0 and ty==0)
				dst = cvCloneImage(tmp2);
			else
			{
				dst=cvCreateImage(cvSize(cvRound(scale*src->width),cvRound(scale*src->height)),src->depth,src->nChannels);	
				cvZero( dst );
				dst = imtranslate(tmp2,tx,ty);
				nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
				cvSaveImage(nam,dst);
				opn_cnt++;

			}
		}
		else
		{
			if(tx==0 and ty==0)
				tmp2 = cvCloneImage(tmp);
			else
			{
				tmp2=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);	
				cvZero( tmp2 );
				tmp2 = imtranslate(tmp,tx,ty);
				nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
				cvSaveImage(nam,tmp2);
				opn_cnt++;
			}
			if(scale==1)
				dst = cvCloneImage(tmp2);
			else
			{
				dst=cvCreateImage(cvSize(cvRound(scale*tmp2->width),cvRound(scale*tmp2->height)),tmp2->depth,tmp2->nChannels);	
				cvZero( dst );
				cvResize(tmp2,dst,intrp[interp-1]);

				nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
				cvSaveImage(nam,dst);
				opn_cnt++;
			}
		}
	}
	else if(o3==1)
	{
		if(tx==0 and ty==0)
			tmp = cvCloneImage(src);
		else
		{
			tmp=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);	
			cvZero( tmp );
			tmp = imtranslate(src,tx,ty);
			nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
			cvSaveImage(nam,tmp);
			opn_cnt++;
//				cout<<opn_cnt<<endl;
		}
		if(o1==2 and o2==3)
		{
			if(scale == 0)
				tmp2 = cvCloneImage(tmp);
			else
			{
				tmp2=cvCreateImage(cvSize(cvRound(scale*tmp->width),cvRound(scale*tmp->height)),tmp->depth,tmp->nChannels);	
				cvZero( tmp2 );
				cvResize(tmp,tmp2,intrp[interp-1]);
				nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
				cvSaveImage(nam,tmp2);
				opn_cnt++;
			}

			if(angl==0)
				dst = cvCloneImage(tmp2);
			else
			{

				dst=cvCreateImage(cvSize(cvRound(scale*tmp->width),cvRound(scale*tmp->height)),tmp->depth,tmp->nChannels);	
				cvZero( dst );
				dst = imscale_rotate(tmp2,angl,1,flag[interp-1]);
				nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
				cvSaveImage(nam,dst);
				opn_cnt++;
			}
		}
		else
		{
			if(angl==0)
				tmp2 = cvCloneImage(tmp);
			else
			{
				tmp2=cvCreateImage(cvSize(tmp->width,tmp->height),tmp->depth,tmp->nChannels);	
				cvZero( tmp2 );
				tmp2 = imscale_rotate(tmp,angl,1,flag[interp-1]);
				nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
				cvSaveImage(nam,tmp2);
				opn_cnt++;
//				cout<<opn_cnt<<endl;
			}

			if(scale==1)
				dst = cvCloneImage(tmp2);
			else
			{
				dst=cvCreateImage(cvSize(cvRound(scale*tmp->width),cvRound(scale*tmp->height)),tmp->depth,tmp->nChannels);	
				cvZero( dst );
				cvResize(tmp2,dst,intrp[interp-1]);
				nt = sprintf(nam,"%s_%d.%s",img_nm,opn_cnt,img_typ);
				cvSaveImage(nam,dst);
				opn_cnt++;
//				cout<<opn_cnt<<endl;
			}
		}

	}
	//		cout<<src->width<<" x "<<src->height<<endl;
	//		cout<<dst->width<<" x "<<dst->height<<endl;

	char nm[50];
	int n = sprintf(nm,"%s.%s",img_nm,img_typ);
	cvSaveImage(nm,dst);

	cvReleaseImage( &src );
	cvReleaseImage( &tmp );
	cvReleaseImage( &tmp2 );
	cvReleaseImage( &dst );

	cout<<opn_cnt-1<<endl;

	return opn_cnt -1;
}

