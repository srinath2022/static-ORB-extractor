#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <queue>
#include <cmath> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "frame.h"

class StaticORBExtractor{
	int MAX_FEATURES;
	double DYNAMIC_DIST_THRESHOLD;
	int nframes; // number of previous frames to use for dynamic point estimation
	queue<Frame*> prev_frames;

public:
	StaticORBExtractor(int nframes, int MAX_FEATURES, double DYNAMIC_DIST_THRESHOLD);
	Frame* extract(const cv::Mat &im);
};