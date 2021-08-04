#include "ImageLoader.h"
#include <opencv2/highgui.hpp>
//#include <ctime>
#include "Utility.h"


void ImageLoader::loadImage()
{
	cout << "Please input the path of the file that including the images' path...\n";
	cout << "(请输入包含图片路径的文件所在路径...)\n";
	cout << "********************************************************************\n";

	cin >> path;

	cout << "********************************************************************\n";
	load();
}

void ImageLoader::load()
{
	cout << "\nLoading...\n";

	fstream imgSrc(path, ios::in);
	vector<string> imgList;
	string s;
	while (imgSrc >> s)
	{
		imgList.push_back(s);
	}

	if (imgList.empty())
	{
		cerr << "No image has been chosen, please reset the file path or check the path is right!\n";
		return;
	}

	Mat temp;
	for (int i = 0; i < imgList.size(); i++)
	{
		temp = imread(imgList[i]);
		imresize(temp, 480);
		imgBuffer.push_back(temp);
	}
	cout << imgList.size() << " images has been chosen\n";
	cout << "image loading completed!\n\n";
}

vector<Mat> ImageLoader::getImgBuffer() const
{
	return imgBuffer;
}

string ImageLoader::getPath() const
{
	return path;
}