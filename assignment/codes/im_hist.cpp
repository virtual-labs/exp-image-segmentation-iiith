#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat adapthist(const Mat &pln, int arg3, int arg4)
{
    int c = pln.cols;
    int r = pln.rows;

    Mat p4 = Mat::zeros(pln.size(), CV_8UC1);
    Mat temp = Mat::zeros(arg3, arg4, CV_8UC1);

    for (int ii = 0; ii < r; ii += arg3)
    {
        for (int jj = 0; jj < c; jj += arg4)
        {
            for (int i = 0; i < arg3 && i + ii < r; i++)
            {
                for (int j = 0; j < arg4 && j + jj < c; j++)
                {
                    temp.at<uchar>(i, j) = pln.at<uchar>(ii + i, jj + j);
                }
            }

            equalizeHist(temp, temp);

            for (int i = 0; i < arg3 && i + ii < r; i++)
            {
                for (int j = 0; j < arg4 && j + jj < c; j++)
                {
                    p4.at<uchar>(ii + i, jj + j) = temp.at<uchar>(i, j);
                }
            }
        }
    }

    return p4;
}

void create_histogram_image(const Mat &bin_img, Mat &hist_img, int bins)
{
    int fc = 256 / bins;
    int hist_size = 256 / fc;
    float range[] = {0, static_cast<float>(256 / fc)};
    const float *ranges[] = {range};
    Mat hist;

    calcHist(&bin_img, 1, 0, Mat(), hist, 1, &hist_size, ranges, true, false);

    double max_value;
    minMaxLoc(hist, 0, &max_value);

    hist_img.setTo(Scalar(200,200,200));

    float w_scale = static_cast<float>(hist_img.cols) / hist_size;

    for (int i = 0; i < hist_size; i++)
    {
        rectangle(hist_img, Point(i * w_scale, hist_img.rows),
                  Point((i + 1) * w_scale, hist_img.rows - cvRound(hist.at<float>(i) * hist_img.rows / max_value)),
                  Scalar(0, 255, 255), -1);
    }
}

Mat Pre_Process(const Mat &src, int bins)
{
    int c = src.cols;
    int r = src.rows;

    Mat final_img = Mat::zeros(src.size(), src.type());

    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            int grval = src.at<uchar>(i, j);
            if (bins == 32)
                grval = grval / 8;
            else if (bins == 64)
                grval = grval / 4;
            else if (bins == 128)
                grval = grval / 2;

            final_img.at<uchar>(i, j) = grval;
        }
    }
    return final_img;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        cout << "Usage: " << argv[0] << " <input_image> <output_image> <operation> [<nbins>] [<arg3>] [<arg4>]" << endl;
        return -1;
    }

    int operation = atoi(argv[3]);
    Mat source = imread(argv[1], IMREAD_GRAYSCALE);
    if (source.empty())
    {
        cout << "Error loading image " << argv[1] << endl;
        return -1;
    }

    string output_file = argv[2];
    string img_nm = output_file.substr(0, output_file.find_last_of('.'));
    string img_typ = output_file.substr(output_file.find_last_of('.') + 1);

    int nbins = 256;
    if (argc >= 5)
        nbins = atoi(argv[4]);

    if (operation < 3)
    {
        Mat bin_img;
        if (nbins != 256)
            bin_img = Pre_Process(source, nbins);
        else
            bin_img = source.clone();

        int c = bin_img.cols;
        int r = bin_img.rows;

        if (operation == 1)
        {
            int mn = 300, mx = -1, avg = 0;
            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    int pixel_val = source.at<uchar>(i, j);
                    mn = min(mn, pixel_val);
                    mx = max(mx, pixel_val);
                    avg += pixel_val;
                }
            }
            avg /= (r * c);
            Mat hist_img(300, 300, CV_8UC3);
            create_histogram_image(bin_img, hist_img, nbins);
            imwrite(output_file, hist_img);
            cout << mn << " " << mx << " " << avg << endl;
            return 0;
        }

        // Process for operation 2
        vector<Mat> bins(4, Mat(r / 2, c / 2, CV_8UC1));
        vector<int> mn(4, 300), mx(4, -1), avg(4, 0);
        int bin_idx;

        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                int bin_i = (i >= r / 2) ? 2 : 0;
                int bin_j = (j >= c / 2) ? 1 : 0;
                bin_idx = bin_i + bin_j;

                int pixel_val = source.at<uchar>(i, j);
                bins[bin_idx].at<uchar>(i % (r / 2), j % (c / 2)) = bin_img.at<uchar>(i, j);
                mn[bin_idx] = min(mn[bin_idx], pixel_val);
                mx[bin_idx] = max(mx[bin_idx], pixel_val);
                avg[bin_idx] += pixel_val;
            }
        }

        for (int i = 0; i < 4; i++)
        {
            avg[i] = 4 * avg[i] / (r * c);
            Mat hist_img(300, 300, CV_8UC3);
            create_histogram_image(bins[i], hist_img, nbins);
            imwrite(img_nm + to_string(i + 1) + "." + img_typ, hist_img);
        }

        cout << mn[0] << " " << mx[0] << " " << avg[0] << " " << mn[1] << " " << mx[1] << " " << avg[1] << " "
             << mn[2] << " " << mx[2] << " " << avg[2] << " " << mn[3] << " " << mx[3] << " " << avg[3] << endl;
        return 0;
    }
    else
    {
        Mat final_img;
        if (operation == 4 && argc >= 6)
        {
            final_img = adapthist(source, atoi(argv[4]), atoi(argv[5]));
        }
        else if (operation == 3)
        {
            equalizeHist(source, final_img);
        }
        else if (operation == 5 && argc >= 5)
        {
            int bin = atoi(argv[4]);
            final_img = source.clone();
            for (int i = 0; i < final_img.rows; i++)
            {
                for (int j = 0; j < final_img.cols; j++)
                {
                    final_img.at<uchar>(i, j) = bin * ((int)(source.at<uchar>(i, j) / bin)) + 2;
                }
            }
        }

        imwrite(output_file, final_img);
        return 0;
    }
}
