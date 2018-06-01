#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

namespace
{

    Mat dst, detected_edges,edges,src,src_gray2, src_gray,standard_hough;

    // windows and trackbars name
    string windowName = "Hough Circle Detection";
    string standard_name = "Standard Hough Lines";
    string standard_name1= "rho";
    string standard_name2 = "theta";
    string cannyThresholdTrackbarName = "Canny threshold for circles";
    string accumulatorThresholdTrackbarName = "Accumulator Threshold";
    string CannyEdgeThreshold = "Canny Threshold for lines";

    // initial and max values of the parameters of interests.
    int cannyThresholdInitialValue = 300;
    int accumulatorThresholdInitialValue = 20;
    int maxAccumulatorThreshold = 50;
    int maxCannyThreshold = 500;
    int ratio = 3;
    int kernel_size = 3;
    int lowThreshold1=280;
    //hough lines specific:
    int s_trackbar1 = 50;
    int min_threshold = 326;
    int rho1=27;
    int max_rho=150;
    int theta1=3;
    int max_theta=300;
    int max_lowThreshold1 = 300;
    int max_trackbar1 = 800;
    bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2,
                          Point2f &r)
    {
        Point2f x = o2 - o1;
        Point2f d1 = p1 - o1;
        Point2f d2 = p2 - o2;

        float cross = d1.x*d2.y - d1.y*d2.x;
        if (abs(cross) < /*EPS*/1e-8)
            return false;

        double t1 = (x.x * d2.y - x.y * d2.x)/cross;
        r = o1 + d1 * t1;
        return true;
    }
    void HoughDetection(const Mat& src_gray2, const Mat& src_display, int cannyThreshold, int accumulatorThreshold,int s_trackbar,int rho,int theta,int lowThreshold)
    {
        // will hold the results of the detection
        std::vector<Vec3f> circles;
        // runs the actual detection of hough circles
        HoughCircles( src_gray2, circles, HOUGH_GRADIENT, 1, src_gray2.rows/8, cannyThreshold, accumulatorThreshold, 0, 0 );

        // clone the colour, input image for displaying purposes
        Mat display = src_display.clone();
        for( size_t i = 0; i < circles.size(); i++ )
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // circle center
            circle( display, center, 6, Scalar(0,255,0), -1, 8, 0 );
            // circle outline
            circle( display, center, radius, Scalar(0,255,0), 3, 8, 0 );
        }

        imwrite("/users/srikanthreddy/desktop/test/src/test1.jpg", display);
        // shows the results
        imshow( windowName, display);
        //process initiated for hough lines
        Canny( src_gray, edges,lowThreshold, lowThreshold*ratio, kernel_size);
        vector<Vec2f> lines;
        cvtColor( edges, standard_hough, COLOR_GRAY2BGR );
        /// Use Standard Hough Transform
        HoughLines( edges, lines, double(rho),double(CV_PI/theta), min_threshold + s_trackbar, 0,0);
        //iteratively search for points and return them
        vector<Point2f> points(2*lines.size());
                for( size_t i = 0; i < lines.size(); i++ )
                {
                    float rho = lines[i][0], theta = lines[i][1];
                    double a = cos(theta), b = sin(theta);
                    double x0 = a*rho, y0 = b*rho;
                    points[2*i].x = cvRound(x0 + 1000*(-b));
                    points[2*i].y = cvRound(y0 + 1000*(a));
                    points[2*i+1].x = cvRound(x0 - 1000*(-b));
                    points[2*i+1].y = cvRound(y0 - 1000*(a));
                    //line( standard_hough,points[2*i], points[2*i+1], Scalar(0,0,255), 3, CV_AA);
                }

               //imwrite("/users/srikanthreddy/desktop/test/src/test.jpg",standard_hough);
                Point2f inter;
                //find intersection
                intersection(points[0],points[1],points[2],points[3],inter);
                //fill color in the respective positions
                Mat img1 = imread("/users/srikanthreddy/desktop/test/src/test1.jpg");
                vector<Point> pts;
                pts.push_back(Point(points[0].x,points[0].y));
                pts.push_back(Point(points[3].x, points[3].y));
                pts.push_back(Point(inter.x, inter.y));
                fillConvexPoly(img1, pts, Scalar(0,0,255));
                fillConvexPoly(standard_hough, pts, Scalar(0,0,255));
                imshow( standard_name, standard_hough );
                imshow("finalresult",img1);
                imwrite("/users/srikanthreddy/desktop/test/src/final.jpg",img1);
      }
}
int main(int argc, char** argv)
{
    // Read the image
    String imageName("/users/srikanthreddy/desktop/test/src/input.png"); // by default
    if (argc > 1)
    {
       imageName = argv[1];
    }
    src = imread( imageName, IMREAD_COLOR );

    if( src.empty() )
    {
        std::cerr<<"Invalid input image\n";
        return -1;
    }

    // Convert it to gray
    cvtColor( src, src_gray2, COLOR_BGR2GRAY );
    cvtColor( src, src_gray, COLOR_BGR2GRAY );

    // Reduce the noise so we avoid false circle detection
    GaussianBlur( src_gray2, src_gray2, Size(9, 9), 2, 2 );

    //declare and initialize both parameters that are subjects to change
    int cannyThreshold = cannyThresholdInitialValue;
    int accumulatorThreshold = accumulatorThresholdInitialValue;
    int rho = rho1;
    int theta = theta1;
    int s_trackbar = s_trackbar1;
    int lowThreshold = lowThreshold1;

    // create the main window, and attach the trackbars
    namedWindow( windowName, WINDOW_AUTOSIZE );
    namedWindow( standard_name, WINDOW_AUTOSIZE );

    createTrackbar(cannyThresholdTrackbarName, windowName, &cannyThreshold,maxCannyThreshold);
    createTrackbar(accumulatorThresholdTrackbarName, windowName, &accumulatorThreshold, maxAccumulatorThreshold);
    createTrackbar( standard_name, standard_name, &s_trackbar, max_trackbar1);
    createTrackbar( standard_name1, standard_name, &rho, max_rho);
    createTrackbar( standard_name2, standard_name, &theta, max_theta);
    createTrackbar( CannyEdgeThreshold, standard_name, &lowThreshold, max_lowThreshold1);

    // infinite loop to display
    // and refresh the content of the output image
    // until the user presses q or Q
    char key = 0;
    while(key != 'q' && key != 'Q')
    {
        // those parameters cannot be =0
        // so we must check here
        cannyThreshold = std::max(cannyThreshold, 1);
        accumulatorThreshold = std::max(accumulatorThreshold, 1);
        s_trackbar = std::max(s_trackbar, 1);
        rho = std::max(rho, 1);
        theta = std::max(theta, 1);
        lowThreshold = std::max(lowThreshold, 1);

        //runs the detection, and update the display
        HoughDetection(src_gray2, src, cannyThreshold, accumulatorThreshold,s_trackbar,rho,theta,lowThreshold);

        // get user key
        key = (char)waitKey(10);
    }
    return 0;
}
