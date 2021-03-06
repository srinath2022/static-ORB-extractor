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
using namespace std;
using namespace cv;

#include "SORBextractor.h"

void LoadImagesTUM(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

class Visualiser{
public:
	~Visualiser(){destroyAllWindows();}
	void show(ORB_SLAM2::StorageFrame* f){
		Mat im = f->image;
		drawKeypoints(im, f->staticKeyPoints, im, Scalar(255,0,0));
		drawKeypoints(im, f->dynamicKeyPoints, im, Scalar(0,0,255));
		imshow( "SORBE", im);
		waitKey(25);
	}
};

int main(int argc, char*argv[]){
	if(argc != 6){
		cout<<"Usage : ./runner path_to_sequence n_previous_frames dataset THRESHOLD slam_mode"<<endl;
		return -1;
	}

	// declare all necessary variables here.
	string path_to_sequence = string(argv[1]);
	int n_previous_frames = strtol(argv[2],NULL,0);
	string dataset = string(argv[3]);
	double DYNAMIC_DIST_THRESHOLD = stod(argv[4]);
	int slam_mode = strtol(argv[5],NULL,0);

	if(dataset=="TUM"){
		vector<string> vstrImageFilenames;
    	vector<double> vTimestamps;
    	string strFile = path_to_sequence+"/rgb.txt";
    	LoadImagesTUM(strFile, vstrImageFilenames, vTimestamps);
    	int nImages = vstrImageFilenames.size();

    	ORB_SLAM2::SORBextractor* SORBE = new ORB_SLAM2::SORBextractor(n_previous_frames, DYNAMIC_DIST_THRESHOLD, 1000, 1.2, 8, 20, 7);
    	Visualiser* visualiser = new Visualiser();

    	cout<<endl;
    	cout << "Processing Images..." << endl;
    	cout << "Images in the sequence: " << nImages << endl;

    	cv::Mat im;
    	for(int i=0; i<nImages; i++){
    		im = cv::imread(path_to_sequence+"/"+vstrImageFilenames[i],CV_LOAD_IMAGE_UNCHANGED);
    		if(im.empty())
	        {
	            cout << "Failed to load image at: "
	                 << path_to_sequence << "/" << vstrImageFilenames[i] << endl;
	            return -1;
	        }
	        Mat imGray;
			cvtColor(im, imGray, cv::COLOR_BGR2GRAY);
			if(slam_mode){
				vector<KeyPoint> keypoints_s;
				vector<KeyPoint> keypoints_d;
				Mat descriptors;
				(*SORBE)(imGray, cv::Mat(), keypoints_s, descriptors);
				ORB_SLAM2::StorageFrame* f = new ORB_SLAM2::StorageFrame(imGray, keypoints_s, keypoints_d);
				visualiser->show(f);
			}else{
				ORB_SLAM2::StorageFrame* f = SORBE->extract(imGray);
		        if(f != NULL){
		        	visualiser->show(f);
		        }
			}
    	}
	}
	else { throw("unknown dataset"); return -1; }

	return 0;
}

/**
 * This code is specifically for loading Tum dataset video sequence
 * Taken as is from ORB-SLAM2 codebase.
 * */
void LoadImagesTUM(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}