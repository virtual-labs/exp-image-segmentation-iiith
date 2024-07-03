#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        cout << "Usage: " << argv[0] << " <input_image> <output_prefix> <conversion_type>" << endl;
        cout << "Conversion types:" << endl;
        cout << "1: RGB to HSV" << endl;
        cout << "2: RGB to CMY" << endl;
        cout << "3: RGB to YCrCb" << endl;
        cout << "4: RGB split into individual channels" << endl;
        cout << "5: RGB to separate CMY channels" << endl;
        return -1;
    }

    Mat source = imread(argv[1]);
    if (source.empty())
    {
        cout << "Error loading image " << argv[1] << endl;
        return -1;
    }

    vector<Mat> channels(3);

    switch (atoi(argv[3]))
    {
    case 1:
        cvtColor(source, source, COLOR_BGR2HSV);
        split(source, channels);
        break;

    case 2:
        channels[0] = (source.col(0) + source.col(1)) / 2;
        channels[1] = (source.col(1) + source.col(2)) / 2;
        channels[2] = (source.col(0) + source.col(2)) / 2;
        break;

    case 3:
        cvtColor(source, source, COLOR_BGR2YCrCb);
        split(source, channels);
        break;

    case 4:
        split(source, channels);
        break;

    case 5:
        channels[0] = source.col(0);
        channels[1] = source.col(1);
        channels[2] = source.col(2);
        break;

    default:
        cout << "Invalid conversion type" << endl;
        return -1;
    }

    string output_prefix = argv[2];
    string output_suffix[3] = { "1", "2", "3" };
    for (int i = 0; i < 3; ++i)
    {
        string output_name = output_prefix + output_suffix[i] + ".jpg";
        imwrite(output_name, channels[i]);
    }

    return 0;
}
