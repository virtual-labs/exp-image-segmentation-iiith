#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include<iostream>
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



/*	CvScalar s,s1;
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
void create_histogram_image(IplImage* bin_img, IplImage* hist_img,int bins) {
	CvHistogram *hist;

	int fc = 256/bins;
	int hist_size = 256/fc;     
	float range[]={0,256/fc};
	float* ranges[] = { range };
	float max_value = 0.0;
	float w_scale = 0.0;

	//	create array to hold histogram values
	hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);

	//	calculate histogram values 
	cvCalcHist( &bin_img, hist, 0, NULL );

	//	Get the minimum and maximum values of the histogram 
	cvGetMinMaxHistValue( hist, 0, &max_value, 0, 0 );

	//	set height by using maximim value
	cvScale( hist->bins, hist->bins, ((float)hist_img->height)/max_value, 0 );

	//	calculate width
	w_scale = ((float)hist_img->width)/hist_size;

	//	plot the histogram 
	for( int i = 0; i < hist_size; i++ ) {

		cvRectangle( hist_img, cvPoint((int)i*w_scale , hist_img->height),
				cvPoint((int)(i+1)*w_scale, hist_img->height - cvRound(cvGetReal1D(hist->bins,i))),
				CV_RGB(255,255,0), -1, 8, 0 );

	}
}
IplImage* Pre_Process(IplImage* src,int bins)
{
	int c = cvGetSize(src).width;
	int r = cvGetSize(src).height;
	
	IplImage* final_img=cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvZero(final_img);
	for(int i=0;i<r;i++)
	{
		for(int j=0;j<c;j++)
		{
				CvScalar s1,s;
				s1 = cvGet2D(src,i,j);
				int grval = s1.val[0];
				if(bins==32)
					grval = grval/8;
				if(bins==64)
					grval = grval/4;
				if(bins==128)
					grval = grval/2;
//				cout<<grval<<" ";
				s.val[0]=grval;
				cvSet2D(final_img,i,j,s);
		}
	}
	return final_img;
}

int main( int argc, char** argv )
{
	
	if(atoi(argv[3])<3) {

	IplImage *source = cvLoadImage(argv[1],0);
	char *img_nm,*img_typ;
	img_nm = strtok (argv[2],".");
	img_typ = strtok (NULL, ".");
	int nbins = atoi(argv[4]);
	IplImage *bin_img = cvCreateImage(cvSize( source->width, source->height ), IPL_DEPTH_8U, 1 );
	if(nbins!=256)
		bin_img = Pre_Process(source,nbins);
	else
		bin_img = cvCloneImage(source);

	int c = cvGetSize(bin_img).width;
	int r = cvGetSize(bin_img).height;


	if(atoi(argv[3])==1)
	{
		int mn=300,mx=-1,avg=0;
		for(int i=0;i<r;i++)
		{
			for(int j=0;j<c;j++)
			{
				CvScalar s1;
				s1 = cvGet2D(source,i,j);
				if(s1.val[0] < mn)
					mn = s1.val[0];
				if(s1.val[0] > mx)
					mx = s1.val[0];
				avg+=s1.val[0];
			}
		}
		avg = avg/(r*c);
		IplImage *hist_img = cvCreateImage(cvSize(300,300), 8,3);
		cvSet( hist_img, cvScalarAll(200), 0 );
		create_histogram_image(bin_img, hist_img,nbins);
		char nam[30];
		int n = sprintf(nam,"%s.%s",img_nm,img_typ);
		cvSaveImage(nam,hist_img);
		cvReleaseImage( &source );
		cvReleaseImage( &hist_img );
		cvReleaseImage( &bin_img );
		cout<<mn<<" "<<mx<<" "<<avg<<endl;
		return 0;

	}
	
	IplImage *bin_img1 = cvCreateImage(cvSize( source->width/2, source->height/2 ), IPL_DEPTH_8U, 1 );
	int mn4=300,mx4=-1,avg4=0;
	for(int i=0;i<r/2;i++)
	{
		for(int j=0;j<c/2;j++)
		{
				CvScalar s,s1;
				s = cvGet2D(bin_img,i,j);
				s1 = cvGet2D(source,i,j);

				if(s1.val[0] < mn4)
					mn4 = s1.val[0];
				if(s1.val[0] > mx4)
					mx4 = s1.val[0];
				avg4+=s1.val[0];
				cvSet2D(bin_img1,i,j,s);
		}
	}
	avg4 = 4*avg4/(r*c);

	IplImage *hist_img = cvCreateImage(cvSize(300,300), 8, 3);
	cvSet( hist_img, cvScalarAll(200), 0 );
	create_histogram_image(bin_img1, hist_img,nbins);
	char nam[30];
	int n = sprintf(nam,"%s1.%s",img_nm,img_typ);
	cvSaveImage(nam,hist_img);

	IplImage *bin_img2 = cvCreateImage(cvSize( source->width/2, source->height/2 ), IPL_DEPTH_8U, 1 );
	int mn1=300,mx1=-1,avg1=0;
	for(int i=0;i<r/2;i++)
	{
		for(int j=c/2;j<c;j++)
		{
				CvScalar s,s1;
				s = cvGet2D(bin_img,i,j);
				s1 = cvGet2D(source,i,j);
				if(s1.val[0] < mn1)
					mn1 = s1.val[0];
				if(s1.val[0] > mx1)
					mx1 = s1.val[0];
				avg1+=s1.val[0];

				cvSet2D(bin_img2,i,j-c/2,s);
		}
	}
	avg1 = 4*avg1/(r*c);

	hist_img = cvCreateImage(cvSize(300,300), 8, 3);
	cvSet( hist_img, cvScalarAll(200), 0 );
	create_histogram_image(bin_img2, hist_img,nbins);
	n = sprintf(nam,"%s2.%s",img_nm,img_typ);
	cvSaveImage(nam,hist_img);

	IplImage *bin_img3 = cvCreateImage(cvSize( source->width/2, source->height/2 ), IPL_DEPTH_8U, 1 );
	int mn2=300,mx2=-1,avg2=0;
	for(int i=r/2;i<r;i++)
	{
		for(int j=0;j<c/2;j++)
		{
				CvScalar s,s1;
				s = cvGet2D(bin_img,i,j);
				s1 = cvGet2D(source,i,j);
				if(s1.val[0] < mn2)
					mn2 = s1.val[0];
				if(s1.val[0] > mx2)
					mx2 = s1.val[0];
				avg2+=s1.val[0];
				cvSet2D(bin_img3,i-r/2,j,s);
		}
	}
	avg2 = 4*avg2/(r*c);

	hist_img = cvCreateImage(cvSize(300,300), 8, 3);
	cvSet( hist_img, cvScalarAll(200), 0 );
	create_histogram_image(bin_img3, hist_img,nbins);
	n = sprintf(nam,"%s3.%s",img_nm,img_typ);
	cvSaveImage(nam,hist_img);


	IplImage *bin_img4 = cvCreateImage(cvSize( source->width/2, source->height/2 ), IPL_DEPTH_8U, 1 );
	int mn3=300,mx3=-1,avg3=0;
	for(int i=r/2;i<r;i++)
	{
		for(int j=c/2;j<c;j++)
		{
				CvScalar s,s1;
				s = cvGet2D(bin_img,i,j);
				s1 = cvGet2D(source,i,j);
				if(s1.val[0] < mn3)
					mn3 = s1.val[0];
				if(s1.val[0] > mx3)
					mx3 = s1.val[0];
				avg3+=s1.val[0];
				cvSet2D(bin_img4,i-r/2,j-c/2,s);
		}
	}
	avg3 = 4*avg3/(r*c);

	hist_img = cvCreateImage(cvSize(300,300), 8,3);
	cvSet( hist_img, cvScalarAll(200), 0 );
	create_histogram_image(bin_img4, hist_img,nbins);
	n = sprintf(nam,"%s4.%s",img_nm,img_typ);
	cvSaveImage(nam,hist_img);

	cvReleaseImage( &source );
	cvReleaseImage( &hist_img );
	cvReleaseImage( &bin_img );
	cvReleaseImage( &bin_img1 );
	cvReleaseImage( &bin_img2 );
	cvReleaseImage( &bin_img3 );
	cvReleaseImage( &bin_img4 );

	cout<<mn4<<" "<<mx4<<" "<<avg4<<" "<<mn1<<" "<<mx1<<" "<<avg1<<" "<<mn2<<" "<<mx2<<" "<<avg2<<" "<<mn3<<" "<<mx3<<" "<<avg3<<" "<<endl;
	return 0;
	} else {
		IplImage* source=cvLoadImage(argv[1],0);

	int c=cvGetSize(source).width;
	int r=cvGetSize(source).height;

	IplImage* final_img=cvCreateImage(cvSize(c,r),IPL_DEPTH_8U,1);

	if (atoi(argv[3])==4)
		final_img=adapthist(source,atoi(argv[4]),atoi(argv[5]));
	else if (atoi(argv[3])==3)
		cvEqualizeHist(source,final_img);
	else if (atoi(argv[3])==5) {
		CvScalar s;	
		int bin=atoi(argv[4]);

		for (int i=0;i<r;i++)
		{
			for (int j=0;j<c;j++)
			{
				s=cvGet2D(source,i,j);
				s.val[0]=bin*((int)(s.val[0]/bin)) + 2;
				cvSet2D(final_img, i, j, s);
			}
		}
	}

	cvSaveImage(argv[2],final_img);

	cvReleaseImage(&source);
	cvReleaseImage(&final_img);

	return 0;
	}
}
