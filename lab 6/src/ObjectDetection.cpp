#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include "ObjectDetection.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


// It searches for the right position, orientation and scale of the object in the scene based on the good_matches.
void ObjectDetection::localizeInImage(const vector<DMatch>& good_matches,
                    const vector<KeyPoint>& keypoints_object,
                    const vector<KeyPoint>& keypoints_scene,
                    const Mat& img_object,
                    const Mat& img_matches)
{
    //-- Localize the object
    vector<Point2f> obj;
    vector<Point2f> scene;
    for (int i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
        scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }

   try {
        Mat H = findHomography(obj, scene, RANSAC);
        //-- Get the corners from the image_1 ( the object to be "detected" )
        vector<Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0, 0);
        obj_corners[1] = cvPoint(img_object.cols, 0);
        obj_corners[2] = cvPoint(img_object.cols, img_object.rows);
        obj_corners[3] = cvPoint(0, img_object.rows);
        vector<Point2f> scene_corners(4);

        perspectiveTransform(obj_corners, scene_corners, H);
        // Draw lines between the corners (the mapped object in the scene - image_2 )
        line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0),
                          scene_corners[1] + Point2f(img_object.cols, 0),
                                             Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0),
                          scene_corners[2] + Point2f(img_object.cols, 0),
                                             Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0),
                          scene_corners[3] + Point2f(img_object.cols, 0),
                                             Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0),
                          scene_corners[0] + Point2f(img_object.cols, 0),
                                             Scalar(0, 255, 0), 4);
         // Draw lines between the corners (the  object to be mapped in the scene - image_2 )
        line(img_matches, obj_corners[0],obj_corners[1], Scalar(0, 0, 255), 4);
        line(img_matches, obj_corners[1],obj_corners[2], Scalar(0, 0, 255), 4);
        line(img_matches, obj_corners[2],obj_corners[3], Scalar(0, 0, 255), 4);
        line(img_matches, obj_corners[3],obj_corners[0], Scalar(0, 0, 255), 4);
   } catch (Exception& e) {}
}

void ObjectDetection::CompAndDetect(Mat img_object,Mat img_scene,string outputFile,int minHessian1)
{
    vector<KeyPoint> keypoints_object, keypoints_scene;//keypoints
    vector<KeyPoint> keypoints_object1, keypoints_scene1;// dummy keypoints to output without keypoints
    Mat descriptors_object, descriptors_scene; // descriptors (features)

    //-- Steps 1 + 2, detect the keypoints and compute descriptors, both in one method
    Ptr<SIFT> sift = SIFT::create( minHessian1 );
    sift->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
    sift->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher; // FLANN - Fast Library for Approximate Nearest Neighbors
    vector< vector< DMatch> > matches;
    matcher.knnMatch( descriptors_object, descriptors_scene, matches, 2 ); // find the best 2 matches of each descriptor

    printf( "object size: %ux%u, scene size: %ux%u\n",img_object.cols, img_object.rows, img_scene.cols, img_scene.rows );

    //-- Step 4: Select only goot matches
    vector< DMatch > good_matches;
    vector< DMatch > bad_matches;//dummy match to hide the lines
    for (int k = 0; k < std::min(descriptors_scene.rows - 1, (int)matches.size()); k++)
    {
        if ( (matches[k][0].distance < 0.6*(matches[k][1].distance)) &&
                ((int)matches[k].size() <= 2 && (int)matches[k].size()>0) )
        {
            // take the first result only if its distance is smaller than 0.6*second_best_dist
            // that means this descriptor is ignored if the second distance is bigger or of similar
            good_matches.push_back( matches[k][0] );
        }
    }


    //-- Step 5: Draw lines between the good matching points
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
            good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
            vector<char>(), DrawMatchesFlags::DEFAULT );
        //dummy image created to use for output images
    drawMatches( img_object, keypoints_object1, img_scene, keypoints_scene1,
            bad_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
            vector<char>(), DrawMatchesFlags::DEFAULT );

    //-- Step 6: Localize the object inside the scene image with a square
    localizeInImage( good_matches, keypoints_object, keypoints_scene, img_object, img_matches );

    //-- Step 7: Show/save matches
    imshow("Object detection", img_matches);
    waitKey(0);
    imwrite(outputFile, img_matches);
}


ObjectDetection::ObjectDetection(string objectInputFile, string sceneInputFile,string outputFile1,int minHessian)
{

    // Load the image from the disk
    Mat img_object = imread( objectInputFile, 1 ); // surf works only with grayscale images
    Mat img_scene = imread( sceneInputFile, 1 );
    string outputFile=outputFile1;
    int minHessian1=minHessian;
    CompAndDetect(img_object,img_scene,outputFile,minHessian1);
}






