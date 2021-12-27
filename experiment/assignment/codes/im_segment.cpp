#include<iostream>
#include<algorithm>
#include<list>
#include<queue>
#include<vector>
#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace std;

int main(int argc, char * argv[])
{
	if(atoi(argv[3])==1)
	{
		IplImage * input = cvLoadImage(argv[1],0);
		int r=cvGetSize(input).height;
		int c=cvGetSize(input).width;
		int num=atoi(argv[4]);
		int thresh1=atoi(argv[5]);
		for(int i=0;i<7;i++) 
		{
			cout<<argv[i]<<"\n";
		}
		CvScalar s;
		if (num==2)
		{
			int thresh2=atoi(argv[6]);
			if(thresh1>thresh2) { 
				num = thresh2;
				thresh2 = thresh1;
				thresh1 = num;
			}
			for (int i=0;i<r;i++)
			{
				for (int j=0;j<c;j++)
				{
					s=cvGet2D(input,i,j);
					if (s.val[0]<thresh1 || s.val[0]>thresh2)
					{
						s.val[0]=0.0;
						cvSet2D(input,i,j,s);
					}
					else
					{
					s.val[0]=255.0;
				    cvSet2D(input,i,j,s);
						}
					
				}
			}
		}
		else
		{
			for (int i=0;i<r;i++)
			{
				for (int j=0;j<c;j++)
				{
					s=cvGet2D(input,i,j);
					if (s.val[0]<thresh1)
					{
						s.val[0]=0.0;
						cvSet2D(input,i,j,s);
					}
					else
					{
					s.val[0]=255.0;
				    cvSet2D(input,i,j,s);
						}
				}
			}
		}
		cvSaveImage(argv[2],input);
		//cvNamedWindow("image",0);
		//cvShowImage("image",input);
		//cvWaitKey(-1);
	}
	else if(atoi(argv[3])==2)
	{
		IplImage * input = cvLoadImage(argv[1],0);
		int r=cvGetSize(input).height;
		int c=cvGetSize(input).width;
		CvScalar s;

		double p[256]={0};
		double P[256]={0};
		double m[256]={0};
		double sigmab=0;
		double mg;
		int kstar=0;
		for (int i=0;i<r;i++)
		{
			for (int j=0;j<c;j++)
			{
				s=cvGet2D(input,i,j);
				p[(int)s.val[0]]+=1;
			}
		}
		int num=r*c;
		p[0]/=num;
		P[0]=p[0];
		for (int i=1;i<=255;i++)
		{
			p[i]/=num;
			P[i]=P[i-1]+p[i];
			m[i]=m[i-1]+i*p[i];
		}
		mg=m[255];
		double maxi=-9999;
		for (int i=0;i<=255;i++)
		{
			sigmab=pow((mg*P[i] - m[i]),2)/(P[i]*(1-P[i]));
			if (maxi<sigmab)
			{
				maxi=sigmab;
				kstar=i;
			}
		}
		int thresh1=kstar;
		for (int i=0;i<r;i++)
		{
			for (int j=0;j<c;j++)
			{
				s=cvGet2D(input,i,j);
				if (s.val[0]<thresh1)
				{
					s.val[0]=0.0;
					cvSet2D(input,i,j,s);
				}
				else
					{
					s.val[0]=255.0;
				    cvSet2D(input,i,j,s);
						}
			}
		}
		cvSaveImage(argv[2],input);
		//cvNamedWindow("image",0);
		//cvShowImage("image",input);
		//cvWaitKey(-1);
	}
	else if(atoi(argv[3])==3)
	{
		//argv[2] == 1 for mean and == 2 for standard deviation
		//argv[3] is the  row
		//argv[4] is the col
		//argv[5] == 1 is all pixels are to be included and == 2 if only 10 pixels are to be included
		//argv[6] is the percentage of error to be allowed in the mean and standard_deviation
		list<int> rq;
		list<int> cq;
		IplImage * input = cvLoadImage(argv[1],0);
		int r=cvGetSize(input).height;
		int c=cvGetSize(input).width;
		IplImage * output = cvCreateImage(cvGetSize(input),IPL_DEPTH_8U,1);
		cvZero(output);
		int A[r][c];
		for(int ii=0;ii<r;ii++)
			for(int jj=0;jj<c;jj++)
				A[ii][jj]=1;
		int type=atoi(argv[4]);
		int i=atoi(argv[5]);
		int j=atoi(argv[6]);
		int include=atoi(argv[7]);
		double percent=atoi(argv[8]);
		double mean=cvGet2D(input,i,j).val[0];
		double  variance=0;
		double std_dev=0;
		double val,val2,temp;
		CvScalar s;
		s.val[0]=255;
		if(type==1)
		{
			val=(1-percent/100)*mean;
			val2=(1+percent/100)*mean;
		}
		else
		{
			val=(mean-std_dev)*(1-percent/100);
			val2=(mean+std_dev)*(1+percent/100);
		}
		rq.push_back(i);
		cq.push_back(j);
		A[r][c]=0;
		list<int>::iterator itr;
		list<int>::iterator itc;
		while(!rq.empty())
		{
			i=rq.front();
			j=cq.front();
			rq.pop_front();
			cq.pop_front();
			if(i+1<r && A[i+1][j])
			{
				A[i+1][j]=0;
				temp=cvGet2D(input,i+1,j).val[0];
				if (temp>=val && temp<=val2)
				{
					rq.push_back(i+1);
					cq.push_back(j);
					cvSet2D(output,i+1,j,s);
				}
			}
			if(i-1>=0 && A[i-1][j])
			{
				A[i-1][j]=0;
				temp=cvGet2D(input,i-1,j).val[0];
				if (temp>=val && temp<=val2)
				{
					rq.push_back(i-1);
					cq.push_back(j);
					cvSet2D(output,i-1,j,s);
				}
			}
			if(j+1<c && A[i][j+1])
			{
				A[i][j+1]=0;
				temp=cvGet2D(input,i,j+1).val[0];
				if (temp>=val && temp<=val2)
				{
					rq.push_back(i);
					cq.push_back(j+1);
					cvSet2D(output,i,j+1,s);
				}
			}
			if(j-1>=0 && A[i][j-1])
			{
				A[i][j-1]=0;
				temp=cvGet2D(input,i,j-1).val[0];
				if (temp>=val && temp<=val2)
				{
					rq.push_back(i);
					cq.push_back(j-1);
					cvSet2D(output,i,j-1,s);
				}
			}
			if(i+1<r && j+1<c && A[i+1][j+1])
			{
				A[i+1][j+1]=0;
				temp=cvGet2D(input,i+1,j+1).val[0];
				if (temp>=val && temp<=val2)
				{
					rq.push_back(i+1);
					cq.push_back(j+1);
					cvSet2D(output,i+1,j+1,s);
				}
			}
			if(i-1>=0 && j-1>=0 && A[i-1][j-1])
			{
				A[i-1][j-1]=0;
				temp=cvGet2D(input,i-1,j-1).val[0];
				if (temp>=val && temp<=val2)
				{
					rq.push_back(i-1);
					cq.push_back(j-1);
					cvSet2D(output,i-1,j-1,s);
				}
			}
			if(j+1<c && i-1>=0 && A[i-1][j+1])
			{
				A[i-1][j+1]=0;
				temp=cvGet2D(input,i-1,j+1).val[0];
				if (temp>=val && temp<=val2)
				{
					rq.push_back(i-1);
					cq.push_back(j+1);
					cvSet2D(output,i-1,j+1,s);
				}
			}
			if(j-1>0 && i+1<r && A[i][j-1])
			{
				A[i+1][j-1]=0;
				temp=cvGet2D(input,i+1,j-1).val[0];
				if (temp>=val && temp<=val2)
				{
					rq.push_back(i+1);
					cq.push_back(j-1);
					cvSet2D(output,i+1,j-1,s);
				}
			}
			

			if(include==1)
			{
				itr=rq.begin();
				itc=cq.begin();
				mean=0;
				variance=0;
				int ii=0;
				for (ii=0;itr!=rq.end();ii++)
				{
					mean+=cvGet2D(input,*itr,*itc).val[0];
					itr++;
					itc++;
				}
				mean/=ii;
				itr=rq.begin();
				itc=cq.begin();
				for (ii=0;ii<10 && itr!=rq.end();ii++)
				{
					variance+=pow(mean-cvGet2D(input,*itr,*itc).val[0],2);
					itr++;
					itc++;
				}
				variance/=ii;
				std_dev=sqrt(variance);
			}
			else if(include==2)
			{
				itr=rq.begin();
				itc=cq.begin();
				mean=0;
				variance=0;
				int ii=0;
				for (ii=0;ii<10 && itr!=rq.end();ii++)
				{
					mean+=cvGet2D(input,*itr,*itc).val[0];
					itr++;
					itc++;
				}
				mean/=ii;
				itr=rq.begin();
				itc=cq.begin();
				for (ii=0;ii<10 && itr!=rq.end();ii++)
				{
					variance+=pow(mean-cvGet2D(input,*itr,*itc).val[0],2);
					itr++;
					itc++;
				}
				variance/=ii;
				std_dev=sqrt(variance);
			}
			if(type==1)
			{
				val=(1-percent/100)*mean;
				val2=(1+percent/100)*mean;
			}
			else
			{
				val=(mean-std_dev)*(1-percent/100);
				val2=(mean+std_dev)*(1+percent/100);
			}
		}
		//cvNamedWindow("regiongrowing",0);
		//cvShowImage("regiongrowing",output);
		//cvWaitKey(-1);

		cvSaveImage(argv[2],output);
		cvReleaseImage(&input);
		cvReleaseImage(&output);
	}

	return 0;
}
