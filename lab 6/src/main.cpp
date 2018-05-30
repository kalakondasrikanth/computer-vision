#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include <ObjectDetection.h>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


int main()
{
        String dir ="/users/srikanthreddy/desktop/from_qt_creator/data";
        for(int k=1;k<=dir.size();k++)
        {
        vector<String> objectInputFile;
        vector<String> sceneInputFile;
        String obj_link="/users/srikanthreddy/desktop/from_qt_creator/data/dataset"+to_string(k)+"/obj*.png";
        String scene_link="/users/srikanthreddy/desktop/from_qt_creator/data/dataset"+to_string(k)+"/scene*.png";
        glob(obj_link,objectInputFile,false);
        glob(scene_link,sceneInputFile,false);
        for(int j=0;j<sceneInputFile.size();j++)
            {
            for(int i=0;i<objectInputFile.size();i++)
                {
                String outputFile1="/users/srikanthreddy/desktop/from_qt_creator/data/dataset"+to_string(k)+"/"+to_string(i)+".png";
                ObjectDetection od (objectInputFile[i], sceneInputFile[j],outputFile1, 400);
                 }
             }
        }
    return 0;
}



