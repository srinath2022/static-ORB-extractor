
#include "frame.h"

using namespace std;
using namespace cv;

Frame::Frame(cv::Mat image, vector<KeyPoint> staticKeyPoints, vector<KeyPoint> dynamicKeyPoints){
	this->image = image;
	this->staticKeyPoints = staticKeyPoints;
	this->dynamicKeyPoints = dynamicKeyPoints;
}