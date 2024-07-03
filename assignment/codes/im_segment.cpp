#include <iostream>
#include <algorithm>
#include <list>
#include <queue>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char * argv[])
{
    if (atoi(argv[3]) == 1)
    {
        Mat input = imread(argv[1], IMREAD_GRAYSCALE);
        if (input.empty())
        {
            cout << "Error: Image not found!" << endl;
            return -1;
        }

        int r = input.rows;
        int c = input.cols;
        int num = atoi(argv[4]);
        int thresh1 = atoi(argv[5]);

        for (int i = 0; i < 7; i++)
        {
            cout << argv[i] << "\n";
        }

        if (num == 2)
        {
            int thresh2 = atoi(argv[6]);
            if (thresh1 > thresh2)
            {
                swap(thresh1, thresh2);
            }

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    uchar pixel = input.at<uchar>(i, j);
                    if (pixel < thresh1 || pixel > thresh2)
                    {
                        input.at<uchar>(i, j) = 0;
                    }
                    else
                    {
                        input.at<uchar>(i, j) = 255;
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    uchar pixel = input.at<uchar>(i, j);
                    if (pixel < thresh1)
                    {
                        input.at<uchar>(i, j) = 0;
                    }
                    else
                    {
                        input.at<uchar>(i, j) = 255;
                    }
                }
            }
        }

        imwrite(argv[2], input);
        //imshow("image", input);
        //waitKey(0);
    }
    else if (atoi(argv[3]) == 2)
    {
        Mat input = imread(argv[1], IMREAD_GRAYSCALE);
        if (input.empty())
        {
            cout << "Error: Image not found!" << endl;
            return -1;
        }

        int r = input.rows;
        int c = input.cols;

        double p[256] = {0};
        double P[256] = {0};
        double m[256] = {0};
        double sigmab = 0;
        double mg;
        int kstar = 0;

        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                uchar pixel = input.at<uchar>(i, j);
                p[pixel]++;
            }
        }

        int num = r * c;
        p[0] /= num;
        P[0] = p[0];
        for (int i = 1; i <= 255; i++)
        {
            p[i] /= num;
            P[i] = P[i - 1] + p[i];
            m[i] = m[i - 1] + i * p[i];
        }
        mg = m[255];
        double maxi = -9999;
        for (int i = 0; i <= 255; i++)
        {
            sigmab = pow((mg * P[i] - m[i]), 2) / (P[i] * (1 - P[i]));
            if (maxi < sigmab)
            {
                maxi = sigmab;
                kstar = i;
            }
        }
        int thresh1 = kstar;
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                uchar pixel = input.at<uchar>(i, j);
                if (pixel < thresh1)
                {
                    input.at<uchar>(i, j) = 0;
                }
                else
                {
                    input.at<uchar>(i, j) = 255;
                }
            }
        }

        imwrite(argv[2], input);
        //imshow("image", input);
        //waitKey(0);
    }
    else if (atoi(argv[3]) == 3)
    {
        // argv[2] == 1 for mean and == 2 for standard deviation
        // argv[3] is the row
        // argv[4] is the col
        // argv[5] == 1 if all pixels are to be included and == 2 if only 10 pixels are to be included
        // argv[6] is the percentage of error to be allowed in the mean and standard deviation
        list<int> rq;
        list<int> cq;
        Mat input = imread(argv[1], IMREAD_GRAYSCALE);
        if (input.empty())
        {
            cout << "Error: Image not found!" << endl;
            return -1;
        }

        int r = input.rows;
        int c = input.cols;
        Mat output = Mat::zeros(input.size(), CV_8U);
        vector<vector<int>> A(r, vector<int>(c, 1));

        int type = atoi(argv[4]);
        int i = atoi(argv[5]);
        int j = atoi(argv[6]);
        int include = atoi(argv[7]);
        double percent = atoi(argv[8]);

        double mean = input.at<uchar>(i, j);
        double variance = 0;
        double std_dev = 0;
        double val, val2, temp;

        Scalar s(255);
        if (type == 1)
        {
            val = (1 - percent / 100) * mean;
            val2 = (1 + percent / 100) * mean;
        }
        else
        {
            val = (mean - std_dev) * (1 - percent / 100);
            val2 = (mean + std_dev) * (1 + percent / 100);
        }

        rq.push_back(i);
        cq.push_back(j);
        A[i][j] = 0;

        while (!rq.empty())
        {
            i = rq.front();
            j = cq.front();
            rq.pop_front();
            cq.pop_front();

            if (i + 1 < r && A[i + 1][j])
            {
                A[i + 1][j] = 0;
                temp = input.at<uchar>(i + 1, j);
                if (temp >= val && temp <= val2)
                {
                    rq.push_back(i + 1);
                    cq.push_back(j);
                    output.at<uchar>(i + 1, j) = 255;
                }
            }
            if (i - 1 >= 0 && A[i - 1][j])
            {
                A[i - 1][j] = 0;
                temp = input.at<uchar>(i - 1, j);
                if (temp >= val && temp <= val2)
                {
                    rq.push_back(i - 1);
                    cq.push_back(j);
                    output.at<uchar>(i - 1, j) = 255;
                }
            }
            if (j + 1 < c && A[i][j + 1])
            {
                A[i][j + 1] = 0;
                temp = input.at<uchar>(i, j + 1);
                if (temp >= val && temp <= val2)
                {
                    rq.push_back(i);
                    cq.push_back(j + 1);
                    output.at<uchar>(i, j + 1) = 255;
                }
            }
            if (j - 1 >= 0 && A[i][j - 1])
            {
                A[i][j - 1] = 0;
                temp = input.at<uchar>(i, j - 1);
                if (temp >= val && temp <= val2)
                {
                    rq.push_back(i);
                    cq.push_back(j - 1);
                    output.at<uchar>(i, j - 1) = 255;
                }
            }
            if (i + 1 < r && j + 1 < c && A[i + 1][j + 1])
            {
                A[i + 1][j + 1] = 0;
                temp = input.at<uchar>(i + 1, j + 1);
                if (temp >= val && temp <= val2)
                {
                    rq.push_back(i + 1);
                    cq.push_back(j + 1);
                    output.at<uchar>(i + 1, j + 1) = 255;
                }
            }
            if (i - 1 >= 0 && j - 1 >= 0 && A[i - 1][j - 1])
            {
                A[i - 1][j - 1] = 0;
                temp = input.at<uchar>(i - 1, j - 1);
                if (temp >= val && temp <= val2)
                {
                    rq.push_back(i - 1);
                    cq.push_back(j - 1);
                    output.at<uchar>(i - 1, j - 1) = 255;
                }
            }
            if (j + 1 < c && i - 1 >= 0 && A[i - 1][j + 1])
            {
                A[i - 1][j + 1] = 0;
                temp = input.at<uchar>(i - 1, j + 1);
                if (temp >= val && temp <= val2)
                {
                    rq.push_back(i - 1);
                    cq.push_back(j + 1);
                    output.at<uchar>(i - 1, j + 1) = 255;
                }
            }
            if (j - 1 > 0 && i + 1 < r && A[i + 1][j - 1])
            {
                A[i + 1][j - 1] = 0;
                temp = input.at<uchar>(i + 1, j - 1);
                if (temp >= val && temp <= val2)
                {
                    rq.push_back(i + 1);
                    cq.push_back(j - 1);
                    output.at<uchar>(i + 1, j - 1) = 255;
                }
            }

            if (include == 1)
            {
                auto itr = rq.begin();
                auto itc = cq.begin();
                mean = 0;
                variance = 0;
                int ii = 0;
                for (ii = 0; itr != rq.end(); ii++)
                {
                    mean += input.at<uchar>(*itr, *itc);
                    ++itr;
                    ++itc;
                }
                mean /= ii;
                itr = rq.begin();
                itc = cq.begin();
                for (ii = 0; ii < 10 && itr != rq.end(); ii++)
                {
                    variance += pow(mean - input.at<uchar>(*itr, *itc), 2);
                    ++itr;
                    ++itc;
                }
                variance /= ii;
                std_dev = sqrt(variance);
            }
            else if (include == 2)
            {
                auto itr = rq.begin();
                auto itc = cq.begin();
                mean = 0;
                variance = 0;
                int ii = 0;
                for (ii = 0; ii < 10 && itr != rq.end(); ii++)
                {
                    mean += input.at<uchar>(*itr, *itc);
                    ++itr;
                    ++itc;
                }
                mean /= ii;
                itr = rq.begin();
                itc = cq.begin();
                for (ii = 0; ii < 10 && itr != rq.end(); ii++)
                {
                    variance += pow(mean - input.at<uchar>(*itr, *itc), 2);
                    ++itr;
                    ++itc;
                }
                variance /= ii;
                std_dev = sqrt(variance);
            }
            if (type == 1)
            {
                val = (1 - percent / 100) * mean;
                val2 = (1 + percent / 100) * mean;
            }
            else
            {
                val = (mean - std_dev) * (1 - percent / 100);
                val2 = (mean + std_dev) * (1 + percent / 100);
            }
        }

        imwrite(argv[2], output);
        //imshow("regiongrowing", output);
        //waitKey(0);
    }

    return 0;
}
