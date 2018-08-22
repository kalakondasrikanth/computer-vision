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
#include "opencv2/ml.hpp"

using namespace std;
using namespace cv;
using namespace ml;
using namespace cv::xfeatures2d;

int SVMtesttrain::trainntest(String test)
{
    //The total number of samples
    const int classSum = 11;
    //Total number of training samples
    const int sampleSum = 5000;
    //The corresponding name for each category
    const std::string labelName[11] = {"100", "10", "90", "70", "50HW", "80","60","40","20","30","50"};
    //Training data and labels
    Mat trainDataMat = Mat::zeros(sampleSum, 1, CV_32FC1);
    Mat labelsMat = Mat::zeros(sampleSum, 1, CV_32SC1);
    int k = 0;
    //training data
    for (int label = 0; label < classSum; label++)
    {
        //Training image folder
        std::string path = "/users/srikanthreddy/desktop/CV/trail2/dataset1/a"+std::to_string(label)+".jpg";
        Mat src = imread(path);
        if (src.empty())
        {
            std::cout<<"can not load image. \n"<<std::endl;
            return -1;
        }
        //imshow("input", src);
        //Get the outline of each image
        std::vector<std::vector<Point> >contours;
        findSamplecontours(src, contours);
        for(int i = 0; i < contours.size(); i++)
        {
            if(contourArea(contours[i]) > 100)
            {
                //Create mask MASK
                Mat mask = Mat::zeros(src.size(), src.type());
                drawContours(mask, contours, i, Scalar(255, 255, 255), -1);

                //Get the image of the MASK corresponding area
                src.copyTo(mask,mask);

                //Find the average of each channel
                Scalar maskSum = sum(mask);
                maskSum = maskSum/contourArea(contours[i]);

                //Take the average of the first three channels, the BGR channel, as a feature
                for (int j = 0; j < 1; j++)
                {
                    trainDataMat.at<float>(k,j) = maskSum[j];
                }
                labelsMat.at<int>(k,0) = label;
                k++;
            }
        }
    }
    std::cout<<"trainDataMat: \n"<<trainDataMat<<"\n"<<std::endl;
    std::cout<<"labelsMat: \n"<<labelsMat<<"\n"<<std::endl;

    //parameters for SVM  and training it.
    Ptr<SVM> model = SVM::create();  
    model->setType(SVM::C_SVC);  
    model->setKernel(SVM::RBF);
    //model->setDegree(1.0);
    // Set parameter C
    model->setC(1);
    // Set parameter Gamma
    model->setGamma(1);
    model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER,100,1e-6)); 
    model->train(trainDataMat,ROW_SAMPLE,labelsMat); 

    //Testing the data
    Mat testImg = imread(test);
    if (testImg.empty())
    {
        std::cout<<"can not load image. \n"<<std::endl;
         return -1;
    }
    std::vector<std::vector<Point> >testContours;
    findSamplecontours(testImg, testContours);
    //Judging the samples in the test image one by one
    for(int i = 0; i < testContours.size(); i++)
    {
        if(contourArea(testContours[i]) > 100)
        {
            Mat testDataMat = Mat::zeros(sampleSum, 1, CV_32FC1);
            Mat testLabelsMat;
            Mat testMask = Mat::zeros(testImg.size(), testImg.type());
            drawContours(testMask, testContours, i, Scalar(255, 255, 255), -1);
            testImg.copyTo(testMask,testMask);

            //Find the average of each channel
            Scalar testMaskSum = sum(testMask);
            testMaskSum = testMaskSum/contourArea(testContours[i]);

            //Take the average of the first three channels, the BGR channel, as a feature
            for (int j = 0; j < 1; j++)
            {
                testDataMat.at<float>(0,j) = testMaskSum[j];
            }
            //Using the trained SVM model for prediction
            model->predict(testDataMat, testLabelsMat);

            //forecast result
            int testLabel = testLabelsMat.at<float>(0,0);
            std::cout <<"testLabel：\n"<<labelName[testLabel]<<std::endl;

            //Draw a forecast result on the test image
            RotatedRect minRect = minAreaRect(testContours[i]);

            //Use the corresponding color rectangle to select the sample box
            rectangle(testImg, minRect.boundingRect(),testMaskSum,2,8);
            putText(testImg, labelName[testLabel],Point(minRect.boundingRect().x,minRect.boundingRect().y),1, 1.5,Scalar(0,255,0),2);
            imshow("test image", testImg);
            waitKey(2000);
        }
    }
     return 0;
}


