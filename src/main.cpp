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

double DYNAMIC_DIST_THRESHOLD = 6.0;

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
	queue<Frame*> prev_frames;

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
		cvtColor(im, imGray, cv::COLOR_BGR2GRAY);
		vector<KeyPoint> keypoints;
  		Mat descriptors;
  		Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
  		orb->detectAndCompute(imGray, Mat(), keypoints, descriptors);

  		vector<KeyPoint> c_staticKeyPoints;
  		vector<KeyPoint> c_dynamicKeyPoints;

  		if(keypoints.size()>1){
  			if(!prev_frames.empty()){
  				// TO-DO: Using only one previous frame for now
  				Frame* prev_f = prev_frames.back();
  				vector<KeyPoint> keypoints_prev;
  				Mat descriptors_prev;
  				orb->detectAndCompute(prev_f->image, Mat(), keypoints_prev, descriptors_prev);

  				// calculate correspondences
  				// TO-DO : Possibly we could use a better algorithm to find faster for all keypoints
  				vector<Point2f> correspondences_in_curr;
  				vector<KeyPoint> kp_correspondences_in_curr;
  				vector<Point2f> correspondences_in_prev;
  				for(int i=0; i<keypoints.size(); i++){
  					// Find the matching correspondence for this descriptor in the previous frame
  					int first_min_index = 0; double first_min_dist = INFINITY;
  					int second_min_index = 0; double second_min_dist = INFINITY;
  					for(int j=0; j<descriptors_prev.rows; j++){
  						double dist = norm(descriptors.row(i),descriptors_prev.row(j),NORM_L2); // Check
  						if(dist<first_min_dist){
  							second_min_dist = first_min_dist;
  							first_min_dist = dist;
  							second_min_index = first_min_index;
  							first_min_index = j;
  						}
  					}
  					if(first_min_dist < 0.8*second_min_dist){
  						correspondences_in_curr.push_back(keypoints[i].pt);
  						kp_correspondences_in_curr.push_back(keypoints[i]);
  						correspondences_in_prev.push_back(keypoints_prev[first_min_index].pt);
  					}
  				}

  				// find the fundamental matrix
  				Mat mask;
  				Mat fundamental_matrix = findFundamentalMat(correspondences_in_curr, correspondences_in_prev, mask, FM_LMEDS);
  				// TO-DO: Use mask to filter the points
  				vector<Point2f> filtered_correspondences_in_curr;
  				vector<KeyPoint> filtered_kp_correspondences_in_curr;
  				vector<Point2f> filtered_correspondences_in_prev;
  				for(int t=0; t<mask.rows; t++){
  					if(mask.at<int>(t,0)){
  						filtered_correspondences_in_curr.push_back(correspondences_in_curr[t]);
  						filtered_kp_correspondences_in_curr.push_back(kp_correspondences_in_curr[t]);
  						filtered_correspondences_in_prev.push_back(correspondences_in_prev[t]);
  					}
  				}

  				vector<Point3f> lines;
  				computeCorrespondEpilines(filtered_correspondences_in_prev, 1, fundamental_matrix, lines);

  				// divide to static and dynamic points
  				for(int k=0; k<filtered_correspondences_in_curr.size(); k++){
  					Point2f point = filtered_correspondences_in_curr[k];
  					Point3f line = lines[k];
  					double point_to_line_dist = abs(line.x*point.x+line.y*point.y+line.z)/sqrt(line.x*line.x+line.y*line.y);
  					if(point_to_line_dist < DYNAMIC_DIST_THRESHOLD){
  						c_staticKeyPoints.push_back(filtered_kp_correspondences_in_curr[k]);
  					}else{
  						c_dynamicKeyPoints.push_back(filtered_kp_correspondences_in_curr[k]);
  					}
  				}
  				
  			}else{
  				// No previous frames, this is the first one. So all are static
  				c_staticKeyPoints = keypoints;
  			}

    		Frame* f = new Frame(imGray, c_staticKeyPoints, c_dynamicKeyPoints);
    		prev_frames.push(f);
    		if(prev_frames.size()>nframes)prev_frames.pop();
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
	if(argc != 5){
		cout<<"Usage : ./main.out path_to_sequence n_previous_frames dataset THRESHOLD"<<endl;
		return -1;
	}

	// declare all necessary variables here.
	string path_to_sequence = string(argv[1]);
	int n_previous_frames = strtol(argv[2],NULL,0);
	string dataset = string(argv[3]);
	DYNAMIC_DIST_THRESHOLD = stod(argv[4]);

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