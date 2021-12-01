#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

class Frame{
public:
	cv::Mat image; // we store gray scale image here
	vector<KeyPoint> staticKeyPoints;
	vector<KeyPoint> dynamicKeyPoints;

	Frame(cv::Mat image, vector<KeyPoint> staticKeyPoints, vector<KeyPoint> dynamicKeyPoints);
};