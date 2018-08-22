#ifndef SVMbased_H
#define SVMbased_H
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/ml.hpp"
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace ml;

class SVMtesttrain{

public:
    void findSamplecontours(InputArray src, OutputArrayOfArrays contours);
    int trainntest(String test);
};

#endif
