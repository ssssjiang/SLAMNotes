#include "FeatureDetector.h"

void saveImage(Mat img, string path)
{
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	try {
		imwrite(path + ".png", img, compression_params);
	}
	catch (cv::Exception& ex) {
		cout << "Exception converting image to PNG format: %s\n" << ex.what();
		return;
	}
}


Mat downSample(Mat img)
{
    Mat smaller;
    smaller.create(img.rows / 2, img.cols / 2, img.type());
    resize(img, smaller, Size(smaller.cols, smaller.rows), 0, 0, CV_INTER_LINEAR);
    return smaller;
}



Mat** buildGaussPyramid(Mat inputImg, int octaves, int intervals, double sig)
{
    Mat** gaussianPyramid = new Mat* [octaves];
    int s = intervals;
    double sigma[intervals + 3];
    double k = pow(2, 1 / s);

    for(int o = 0; o < octaves; o++) 
        gaussianPyramid[o] = new Mat [s + 3];

    for(int i = 0; i < s + 3; i++)
    {
        if(i == 0) sigma[i] = sig;
        else if(i == 1) sigma[i] = sig * sqrt(k * k - 1);
        else sigma[i] = k * sigma[i - 1];
    }

    for(int o = 0; o < octaves; o++)
    {   
        for(int i = 0; i < s + 3; i++)
        {
            if(o == 0 && i == 0) gaussianPyramid[o][i] = inputImg.clone();
            else if(i == 0)
            {
                gaussianPyramid[o][i] = downSample(gaussianPyramid[o - 1][s]);
            }
            else
            {
                //gaussianPyramid[o][i - 1].copyTo(gaussianPyramid[o][i]);
                cvtColor(gaussianPyramid[o][i - 1], gaussianPyramid[o][i], CV_BGR2GRAY);
                GaussianBlur(gaussianPyramid[o][i - 1], gaussianPyramid[o][i], Size(3, 3), sigma[i], sigma[i], BORDER_DEFAULT);
            }
        }
    }
    return gaussianPyramid;
}


Mat** buildDogPyramid(Mat** gp, int octaves, int intervals)
{
    Mat** dogPyramid = new Mat* [octaves];
    for(int o = 0; o < octaves; o++) 
        dogPyramid[o] = new Mat [intervals + 2];

    for(int o = 0; o < octaves; o++)
    {
        for(int i = 0; i < intervals + 2; i++)
        {
            gp[o][i].copyTo(dogPyramid[o][i]);
            subtract(gp[o][i + 1], gp[o][i], dogPyramid[o][i]);
        }
    }
    return dogPyramid;
}


void localExtremaDetect(Mat** dogPyramid)
{

}