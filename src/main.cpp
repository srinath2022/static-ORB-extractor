#include <iostream>
#include <string>
#include <stdlib.h>
#include<fstream>
#include <queue>
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

void LoadImagesTUM(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

class Frame{
public:
	cv::Mat image; // we store gray scale image here
	vector<KeyPoint> staticKeyPoints;
	vector<KeyPoint> dynamicKeyPoints;

	Frame(cv::Mat image, vector<KeyPoint> staticKeyPoints, vector<KeyPoint> dynamicKeyPoints){
		this->image = image;
		this->staticKeyPoints = staticKeyPoints;
		this->dynamicKeyPoints = dynamicKeyPoints;
	}
};

class StaticORBExtractor{
	int MAX_FEATURES;
	int nframes; // number of previous frames to use for dynamic point estimation
	queue<Frame> prev_frames;

public:

	StaticORBExtractor(int nframes, int MAX_FEATURES){
		if(MAX_FEATURES<=0){throw("MAX_FEATURES must be > 0");}
		if(nframes<0){throw("number of previous frames must be >= 0");}
		this->nframes = nframes;
		this->MAX_FEATURES = MAX_FEATURES;
	}

	// reference code :- https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
	Frame* extract(const cv::Mat &im){
		Mat imGray;
		cvtColor(im, imGray, cv::COLOR_BGR2GRAY); // TO-DO: check back on this
		vector<KeyPoint> keypoints;
  		Mat descriptors;
  		Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
  		orb->detectAndCompute(imGray, Mat(), keypoints, descriptors);

  		// TO-DO : Implement actual separator
  		if(keypoints.size()>1){
  			vector<KeyPoint> staticKeypoints(keypoints.begin(), keypoints.begin() + keypoints.size() / 2);
    		vector<KeyPoint> dynamicKeyPoints(keypoints.begin() + keypoints.size() / 2, keypoints.end());
    		Frame* f = new Frame(imGray, staticKeypoints, dynamicKeyPoints);
    		return f;
  		}
  		return NULL;
	}
};

class Visualiser{
public:
	~Visualiser(){destroyAllWindows();}
	void show(Frame* f){
		Mat im = f->image;
		drawKeypoints(im, f->staticKeyPoints, im, Scalar(255,0,0));
		drawKeypoints(im, f->dynamicKeyPoints, im, Scalar(0,0,255));
		imshow( "SORBE", im);
		waitKey(25);
	}
};

int main(int argc, char*argv[]){
	if(argc != 4){
		cout<<"Usage : ./main.out path_to_sequence n_previous_frames dataset"<<endl;
		return -1;
	}

	// declare all necessary variables here.
	string path_to_sequence = string(argv[1]);
	int n_previous_frames = strtol(argv[2],NULL,0);
	string dataset = string(argv[3]);

	if(dataset=="TUM"){
		vector<string> vstrImageFilenames;
    	vector<double> vTimestamps;
    	string strFile = path_to_sequence+"/rgb.txt";
    	LoadImagesTUM(strFile, vstrImageFilenames, vTimestamps);
    	int nImages = vstrImageFilenames.size();

    	StaticORBExtractor* SORBE = new StaticORBExtractor(n_previous_frames, 500);
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
	        Frame* f = SORBE->extract(im);
	        if(f != NULL){
	        	visualiser->show(f);
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