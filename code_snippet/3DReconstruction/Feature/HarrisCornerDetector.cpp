#include <iostream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


// find the maximum value in matirx m
double findMaximum(Mat m)
{
    double mm = 0;
    for(int i = 0; i < m.rows; i++)
    {
        for(int j = 0; j < m.cols; j++)
        {
            mm = max(mm, m.at<double>(i, j));
        }
    }
    return mm;
}


/**
 * Harris corner detector
 * param img: input image to do corner detecting
 * param alpha: 
**/
void runHarrisCornerDetector(Mat img, Mat& imgDst, double alpha)
{
    // convert image to gray scale
    Mat src;
	if (img.channels() == 3)
	{
		cvtColor(img, src, CV_BGR2GRAY);
	}
	else
	{
		src = img.clone();
	}
    src.convertTo(src, CV_64F);

    Mat Ix, Iy;

    Mat xKernel = (Mat_<double>(1,3) << -1, 0, 1);
	Mat yKernel = xKernel.t();

	filter2D(src, Ix, CV_64F, xKernel);
	filter2D(src, Iy, CV_64F, yKernel);

	Mat Ix2,Iy2,Ixy;
	Ix2 = Ix.mul(Ix);
	Iy2 = Iy.mul(Iy);
	Ixy = Ix.mul(Iy);

	Mat gaussKernel = getGaussianKernel(7, 1);
	filter2D(Ix2, Ix2, CV_64F, gaussKernel);
	filter2D(Iy2, Iy2, CV_64F, gaussKernel);
	filter2D(Ixy, Ixy, CV_64F, gaussKernel);

    // corner reponse, according to the equation of the paper of Harris & Stephens, 1988
    Mat responseMat;
    responseMat.create(src.size(), src.type());
    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            double tr = Ix2.at<double>(i, j) + Iy2.at<double>(i, j);
            double det = Ix2.at<double>(i, j) * Iy2.at<double>(i, j) - Ixy.at<double>(i, j) * Ixy.at<double>(i, j);
            responseMat.at<double>(i, j) = det - alpha * tr * tr;
        }
    }

    double maxima = findMaximum(responseMat);
    double threshold = 0.01 * maxima;
	vector<Point2i> points;
	harrisCornerDetector(responseMat, points, threshold);

    Mat dilated;
	Mat localMax;
	dilate(responseMat, dilated, Mat());
	compare(responseMat, dilated, localMax, CMP_EQ);
	

	Mat cornerMap;
	cornerMap = responseMat > threshold;
	bitwise_and(cornerMap, localMax, cornerMap);
	
	imgDst = cornerMap.clone();
}


void drawCornerOnImage(Mat& image, const Mat& binary)
{
    Mat_<uchar>::const_iterator it = binary.begin<uchar>();
    Mat_<uchar>::const_iterator itd = binary.end<uchar>();
    for (int i = 0; it != itd; it++, i++)
    {
        if (*it)
            circle(image, Point(i%image.cols, i / image.cols), 5, Scalar(0, 255, 0), 1);
    }
}


int main()
{
    Mat img = imread("/home/chenyu/图片/david/B00.jpg");
    Mat imgDst;
    runHarrisCornerDetector(img, imgDst, 0.05);
    drawCornerOnImage(img, imgDst);
	imshow("Myharris", img);

    waitKey(0);
}