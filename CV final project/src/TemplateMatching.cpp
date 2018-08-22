#include "TemplateMatching.h"
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

int TemplateMatching::TMatches(String filename, String templatename,String label)
{
    Mat ref = cv::imread(filename);
    Mat tpl = cv::imread(templatename);
    if(ref.empty() || tpl.empty())
    {
        cout << "Error reading file(s)!" << endl;
        return -1;
    }

    Mat gref, gtpl;
    cvtColor(ref, gref, CV_BGR2GRAY);
    cvtColor(tpl, gtpl, CV_BGR2GRAY);

    const int low_canny = 110;
    Canny(gref, gref, low_canny, low_canny*3);
    Canny(gtpl, gtpl, low_canny, low_canny*3);

    //imshow("file", gref);
    //imshow("template", gtpl);

    Mat res_32f(ref.rows - tpl.rows + 1, ref.cols - tpl.cols + 1, CV_32FC1);
    matchTemplate(gref, gtpl, res_32f, CV_TM_CCOEFF_NORMED);

    Mat res;
    res_32f.convertTo(res, CV_8U, 255.0);
   // imshow("result", res);

    int size = ((tpl.cols + tpl.rows) / 4) * 2 + 1; //force size to be odd
    adaptiveThreshold(res, res, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, size, -64);
    //imshow("result_thresh", res);

    while(1)
    {
        double minval, maxval;
        Point minloc, maxloc;
        minMaxLoc(res, &minval, &maxval, &minloc, &maxloc);

        if(maxval > 0)
        {
            rectangle(ref, maxloc, Point(maxloc.x + tpl.cols, maxloc.y + tpl.rows), Scalar(0,255,0), 2);
            putText(ref, label, cvPoint(30,30),
                FONT_HERSHEY_COMPLEX, 1, cvScalar(255,255,255), 1, CV_AA);
            floodFill(res, maxloc, 0); //mark drawn blob
        }
        else
            break;
    }

    imshow("final", ref);
    waitKey(0);
   return 0;
}

