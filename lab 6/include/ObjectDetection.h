#ifndef OBJECTDETECTION_H
#define OBJECTDETECTION_H
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

class ObjectDetection{

public:
    ObjectDetection(string objectInputFile, string sceneInputFile,string outputFile1,int minHessian);
private:
    void localizeInImage(const vector<DMatch>& good_matches,
                        const vector<KeyPoint>& keypoints_object,
                        const vector<KeyPoint>& keypoints_scene,
                        const Mat& img_object,
                        const Mat& img_matches);
    void CompAndDetect(Mat img_object,Mat img_scene,string outputFile,int minHessian1);
};

#endif
