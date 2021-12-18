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

#include "ORBextractor.h"

#ifndef SORBEXTRACTOR_H
#define SORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>

namespace ORB_SLAM2
{

	class StorageFrame
	{
	public:
		cv::Mat image; // we store gray scale image here
		std::vector<cv::KeyPoint> staticKeyPoints;
		std::vector<cv::KeyPoint> dynamicKeyPoints;

		StorageFrame(cv::Mat image, std::vector<cv::KeyPoint> staticKeyPoints, std::vector<cv::KeyPoint> dynamicKeyPoints);
	};

	class SORBextractor
	{
	public:
	    
	    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

	    SORBextractor(int nframes, double DYNAMIC_DIST_THRESHOLD, int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);

	    ~SORBextractor(){}

	    StorageFrame* extract(cv::InputArray im);

	    // Compute the Static ORB features and descriptors on an image.
	    // Mask is ignored in the current implementation.
	    void operator()( cv::InputArray image, cv::InputArray mask,
	      std::vector<cv::KeyPoint>& keypoints,
	      cv::OutputArray descriptors);

	protected:
	    double DYNAMIC_DIST_THRESHOLD;
		int nframes; // number of previous frames to use for dynamic point estimation
		std::queue<StorageFrame*> prev_frames;
		
	public:
        ORBextractor* orbExtractor;

	};
} 

#endif
