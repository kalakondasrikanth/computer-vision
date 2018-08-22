#ifndef TEMPLATEMATCHING_H
#define TEMPLATEMATCHING_H
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

class TemplateMatching{

public:
    int TMatches(String filename, String templatename,String label);

};

#endif
