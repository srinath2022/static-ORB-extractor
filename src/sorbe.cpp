#include "sorbe.h"

using namespace std;
using namespace cv;

StaticORBExtractor::StaticORBExtractor(int nframes, int MAX_FEATURES, double DYNAMIC_DIST_THRESHOLD)
{
	if(MAX_FEATURES<=0){throw("MAX_FEATURES must be > 0");}
	if(nframes<0){throw("number of previous frames must be >= 0");}
	this->nframes = nframes;
	this->MAX_FEATURES = MAX_FEATURES;
	this->DYNAMIC_DIST_THRESHOLD = DYNAMIC_DIST_THRESHOLD;
}

Frame* StaticORBExtractor:: extract(const cv::Mat &im)
{
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