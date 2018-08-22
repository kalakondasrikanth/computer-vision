#include "svmbased.h"
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

void SVMtesttrain::findSamplecontours(InputArray src, OutputArrayOfArrays contours)
{
    Mat dst,blueChannels, redChannels;
    std::vector<Mat>channels;

    //Divide the original image into multiple single-channel images
        split(src, channels);
        blueChannels = channels.at(0);
        redChannels = channels.at(2);

    //Binarize using different thresholds for R and B channels, respectively
        threshold(redChannels, redChannels, 200, 255, THRESH_BINARY_INV);
        threshold(blueChannels, blueChannels, 200, 255, THRESH_BINARY_INV);

    //Merge two binarized regions
        add(blueChannels,redChannels, dst);

    //Close the acquired area and remove the center hole
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5), Point(-1,-1));
        morphologyEx(dst,dst,CV_MOP_CLOSE,kernel);

    //Corrosion operation to remove some transitional edges
       Mat kernel2 = getStructuringElement(MORPH_RECT, Size(5,5), Point(-1,-1));
       erode(dst, dst,kernel2);
       dilate(dst,dst,kernel2);


    //imshow("00", dst);
    //waitKey();
    Canny(dst, dst, 100, 400, 3);
    std::vector<Vec4i>hierarchy;
    //Get contour
    findContours(dst, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0,0));
}
