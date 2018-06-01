#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void p1()
{
    Mat src;
    /// Load image
    src = imread( "lab3_image.jpg" );

    /// Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;
    split( src, bgr_planes );

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

    /// Compute the histograms:
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage_b( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat histImage_g( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat histImage_r( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage_b.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage_g.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage_r.rows, NORM_MINMAX, -1, Mat() );

    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage_b, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
             Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage_g, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
             Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage_r, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
             Scalar( 0, 0, 255), 2, 8, 0  );
    }

    /// Store
    imwrite("downloads/histImage_b", histImage_b);
    imwrite("downloads/histImage_g", histImage_g);
    imwrite("downloads/histImage_r", histImage_r);
    waitKey(0);
}

void p2()
{
    Mat src;
    /// Load image
    src = imread( "lab3_image.jpg" );

    /// Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;

    //split the quilized image
    split( src, bgr_planes );

    //Equilize each channel separately
    equalizeHist(bgr_planes[0],bgr_planes[0]);
    equalizeHist(bgr_planes[1],bgr_planes[1]);
    equalizeHist(bgr_planes[2],bgr_planes[2]);

    // Merge the the color planes back into an Lab image
    merge(bgr_planes, src);

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

    /// Compute the histograms:
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage_b( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat histImage_g( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat histImage_r( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage_b.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage_g.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage_r.rows, NORM_MINMAX, -1, Mat() );

    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage_b, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
             Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage_g, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
             Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage_r, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
             Scalar( 0, 0, 255), 2, 8, 0  );
    }

    /// Store
    imwrite("downloads/histImage_b", histImage_b);
    imwrite("downloads/histImage_g", histImage_g);
    imwrite("downloads/histImage_r", histImage_r);
    waitKey(0);
}

void p3()
{
    Mat src,cvt_src,dst,image_clahe;
    /// Load image
    src = imread( "lab3_image.jpg" );

    // convert RGB image to gray
    cvtColor(src, cvt_src, CV_BGR2Lab);

    // Extract the L channel
    vector<Mat> lab_planes(3);
    split(cvt_src, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);
    clahe->apply(lab_planes[0], dst);

    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    merge(lab_planes, cvt_src);

   // convert back to RGB
   cvtColor(cvt_src, image_clahe, CV_Lab2BGR);

   // Store the results  (you might also want to see lab_planes[0] before and after).
   imwrite("downloads/src.jpg", src);
   imwrite("downloads/image_clahe.jpg", image_clahe);

    waitKey(0);
}

void p4()
{
    cout<<"Please select a valid option !!";
}

int main()
{
    int i;

    cout<< " This program has 3 section...\n"
    "PART 1: SPLITTING the image into BGR planes and diplay its histograms.\n"
    "PART 2: EQUILIZING the  source image and displaying the equilized image and its respective histograms.\n"
    "PART 3: CONVERTING RBG image to LAB colorspace and EQUILIZING L channel.(Basically luminance).\n";

for(int a=1;a<4;a++)
{
    cout<<"Please choose the section you desire to inspect !!" <<endl;
    cin>> i;
    switch (i)
    {
        case 1: p1(); break;
        case 2: p2(); break;
        case 3: p3(); break;
        default: p4();
    }
}
return 0;
}

