#include<iostream>
#include<algorithm>
#include<queue>
#include<vector>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;

IplImage* adapthist(IplImage* pln, int arg3, int arg4)
{
	int c=cvGetSize(pln).width;
	int r=cvGetSize(pln).height;

        IplImage *p4=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);


        IplImage *temp=cvCreateImage(cvSize(arg3,arg4),IPL_DEPTH_8U,1);

	        CvScalar s;
		        for (int ii=0;ii<r;ii+=arg3)
			        {
				                for (int jj=0;jj<c;jj+=arg4)
						                {
								                        for (int i=0;i<arg3 && i+ii<r;i++)
											                        {
														                                for (int j=0;j<arg4 && j+jj<c;j++)
																		                                {
																						                                        s=cvGet2D(pln,ii+i,jj+j);
																											                                        cvSet2D(temp,i,j,s);
																																                                }
																																				                        }

																																							                        cvEqualizeHist(temp,temp);

																																										                        for (int i=0;i<arg3 && i+ii<r;i++)
																																													                        {
																																																                                for (int j=0;j<arg4 && j+jj<c;j++)
																																																				                                {
																																																								                                        s=cvGet2D(temp,i,j);
																																																													                                        cvSet2D(p4,ii+i,jj+j,s);
																																																																		                                }
																																																																						                        }
																																																																									                }
																																																																											        }




/*
	CvScalar s,s1;
        int cdf[256]={0};
        int hst[256]={0};
        int count;
        for (int i=0;i<r;i++)
        {
                for (int j=0;j<c;j++)
                {
                        for (int v=0;v<256;v++)
                                hst[v]=0;
                        int x = i - (int)floor(arg3/2);
                        int y = j - (int)floor(arg4/2);
                        int p = (int)floor(arg3/2);
                        int q = (int)floor(arg4/2);
                        for (int ii = ((0>=x)?0:x); ii<r && ii<=i+p; ii++ )
                        {
                                for (int jj= ((0>=y)?0:y); jj<c && jj<=j+q; jj++ )
                                {
                                        s=cvGet2D(pln,ii,jj);
                                        hst[(int)s.val[0]]++;
                                }
                        }
                        cdf[0]=hst[0];
                        for (int v=1;v<256;v++)
                                cdf[v]=cdf[v-1]+hst[v];
                        s1=cvGet2D(pln,i,j);
                        s.val[0]=floor( 255 * ( cdf[(int)s1.val[0]] ) / ( arg3 * arg4 ) );
                        cvSet2D(p4,i,j,s);
                }
        }
*/
	return p4;
}

// argv[3] will tell whethr normal histeq (1) or adaptive histeq (2)
// argv[5] will tells on which plane the proc. needs to be done --> input should be 1 or 2 or 3
// and argv[4] tells which color model to select --> input should be "1" for RGB, "2" for HSV, and "3" for CMY
// argv[6] and argv[7] tell the kernel size -> LXB

int main(int argc, char* argv[])
{
	IplImage *source=cvLoadImage(argv[1],1);

	int c=cvGetSize(source).width;
	int r=cvGetSize(source).height;

	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,3);
	IplImage* out_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,3);
	IplImage *p1=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
        IplImage *p2=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
        IplImage *p3=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
        IplImage *p4=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
        IplImage *p5=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);
        IplImage *p6=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);

	if (atoi(argv[4])==1)
        {
                CvScalar s,t;
                t.val[0]=0;
                t.val[1]=0;
                t.val[2]=0;
                for (int i=0;i<r;i++)
                {
                        for (int j=0;j<c;j++)
                        {
                                s=cvGet2D(source,i,j);
                                t.val[0]=s.val[0];
                                cvSet2D(p1,i,j,t);
                                t.val[0]=s.val[1];
                                cvSet2D(p2,i,j,t);
                                t.val[0]=s.val[2];
                                cvSet2D(p3,i,j,t);
                        }
                }
                cvMerge(p1,p2,p3,NULL,out_img);
        }
        else if (atoi(argv[4])==2)
        {
                cvCvtColor(source,out_img,CV_RGB2HSV);
                cvSplit(out_img,p1,p2,p3,0);
        }
        else if (atoi(argv[4])==3)
        {
                CvScalar s,t;
                t.val[0]=0;
                t.val[1]=0;
                t.val[2]=0;
                for (int i=0;i<r;i++)
                {
                        for (int j=0;j<c;j++)
                        {
                                s=cvGet2D(source,i,j);
                                t.val[0]=(s.val[2]+s.val[1])/2;
                                cvSet2D(p1,i,j,t);
                                t.val[0]=(s.val[0]+s.val[2])/2;
                                cvSet2D(p2,i,j,t);
                                t.val[0]=(s.val[0]+s.val[1])/2;
                                cvSet2D(p3,i,j,t);
                        }
                }
                cvMerge(p1,p2,p3,NULL,out_img);
        }

	if (atoi(argv[5])==1)
	{
		// Apply on H plane
		if (atoi(argv[3])==2)
			p4=adapthist(p1,atoi(argv[6]),atoi(argv[7]));
		else if (atoi(argv[3])==1)
			cvEqualizeHist(p1,p4);

		cvMerge(p4,p2,p3,NULL,out_img);

		cvSaveImage("p1.jpg",p4);
		cvSaveImage("p2.jpg",p2);
		cvSaveImage("p3.jpg",p3);
	}

	if (atoi(argv[5])==2)
	{
		// Apply on S plane
		if (atoi(argv[3])==2)
			p5=adapthist(p2,atoi(argv[6]),atoi(argv[7]));
		else if (atoi(argv[3])==1)
			cvEqualizeHist(p2,p5);

		cvMerge(p1,p5,p3,NULL,out_img);

		cvSaveImage("p1.jpg",p1);
		cvSaveImage("p2.jpg",p5);
		cvSaveImage("p3.jpg",p3);
	}

	if (atoi(argv[5])==4)
	{
		// Apply on V plane
		if (atoi(argv[3])==2)
			p6=adapthist(p3,atoi(argv[6]),atoi(argv[7]));
		else if (atoi(argv[3])==1)
			cvEqualizeHist(p3,p6);

		cvMerge(p1,p2,p6,NULL,out_img);

		cvSaveImage("p1.jpg",p1);
		cvSaveImage("p2.jpg",p2);
		cvSaveImage("p3.jpg",p6);
	}

	if (atoi(argv[5])==3)
	{
		// Apply on H+S plane
		if (atoi(argv[3])==2)
		{
			p4=adapthist(p1,atoi(argv[6]),atoi(argv[7]));
			p5=adapthist(p2,atoi(argv[6]),atoi(argv[7]));
		}
		else if (atoi(argv[3])==1)
		{
			cvEqualizeHist(p1,p4);
                	cvEqualizeHist(p2,p5);
		}

		cvMerge(p4,p5,p3,NULL,out_img);

		cvSaveImage("p1.jpg",p4);
		cvSaveImage("p2.jpg",p5);
		cvSaveImage("p3.jpg",p3);
	}

	if (atoi(argv[5])==6)
	{
		// Apply on S+V plane
		if (atoi(argv[3])==2)
		{
			p5=adapthist(p2,atoi(argv[6]),atoi(argv[7]));
			p6=adapthist(p3,atoi(argv[6]),atoi(argv[7]));
		}
		else if (atoi(argv[3])==1)
		{
			cvEqualizeHist(p2,p5);
                	cvEqualizeHist(p3,p6);
		}

		cvMerge(p1,p5,p6,NULL,out_img);

		cvSaveImage("p1.jpg",p1);
		cvSaveImage("p2.jpg",p5);
		cvSaveImage("p3.jpg",p6);
	}

	if (atoi(argv[5])==5)
	{
		// Apply on H+V plane
		if (atoi(argv[3])==2)
		{
			p4=adapthist(p1,atoi(argv[6]),atoi(argv[7]));
			p6=adapthist(p3,atoi(argv[6]),atoi(argv[7]));
		}
		else if (atoi(argv[3])==1)
		{
			cvEqualizeHist(p1,p4);
                	cvEqualizeHist(p3,p6);
		}

		cvMerge(p4,p2,p6,NULL,out_img);

		cvSaveImage("p1.jpg",p4);
		cvSaveImage("p2.jpg",p2);
		cvSaveImage("p3.jpg",p6);
	}

	if (atoi(argv[5])==7)
	{
		// Apply on H+S+V plane
		if (atoi(argv[3])==2)
		{
			p4=adapthist(p1,atoi(argv[6]),atoi(argv[7]));
			p5=adapthist(p2,atoi(argv[6]),atoi(argv[7]));
			p6=adapthist(p3,atoi(argv[6]),atoi(argv[7]));
		}
		else if (atoi(argv[3])==1)
		{
			cvEqualizeHist(p1,p4);
			cvEqualizeHist(p2,p5);
			cvEqualizeHist(p3,p6);
		}

		cvMerge(p4,p5,p6,NULL,out_img);

		cvSaveImage("p1.jpg",p4);
		cvSaveImage("p2.jpg",p5);
		cvSaveImage("p3.jpg",p6);
	}


	if (atoi(argv[4])==1)
	{
		CvScalar s,t;
		t.val[0]=0;
		t.val[1]=0;
		t.val[2]=0;
		for (int i=0;i<r;i++)
		{
			for (int j=0;j<c;j++)
			{
				s=cvGet2D(out_img,i,j);
				t.val[0]=s.val[0];
				cvSet2D(p1,i,j,t);
				t.val[0]=s.val[1];
				cvSet2D(p2,i,j,t);
				t.val[0]=s.val[2];
				cvSet2D(p3,i,j,t);
			}
		}
		cvMerge(p1,p2,p3,NULL,final_img);
	}
	else if (atoi(argv[4])==2)
	{
		cvCvtColor(out_img,final_img,CV_HSV2RGB);
		cvSplit(final_img,p1,p2,p3,0);
	}
	else if (atoi(argv[4])==3)
	{
		CvScalar s,t;
		t.val[0]=0;
		t.val[1]=0;
		t.val[2]=0;
		for (int i=0;i<r;i++)
		{
			for (int j=0;j<c;j++)
			{
				s=cvGet2D(out_img,i,j);
				t.val[0]=(-s.val[0]+s.val[1]+s.val[2]);
				cvSet2D(p1,i,j,t);
				t.val[0]=(s.val[0]-s.val[1]+s.val[2]);
				cvSet2D(p2,i,j,t);
				t.val[0]=(s.val[0]+s.val[1]-s.val[2]);
				cvSet2D(p3,i,j,t);
			}
		}
		cvMerge(p1,p2,p3,NULL,final_img);
	}

	cvSaveImage(argv[2],final_img);

	cvReleaseImage(&p1);
        cvReleaseImage(&p2);
        cvReleaseImage(&p3);
        cvReleaseImage(&p4);
        cvReleaseImage(&p5);
        cvReleaseImage(&p6);
	cvReleaseImage(&source);
	cvReleaseImage(&final_img);
	cvReleaseImage(&out_img);

	return 0;
}
