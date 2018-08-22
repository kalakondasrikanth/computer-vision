#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "TemplateMatching.h"
#include "svmbased.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main()
{
    TemplateMatching a;
    SVMtesttrain b;
    String test=("/users/srikanthreddy/desktop/test1/trail2/dataset1/boards.jpg"); //test image for SVM
    b.trainntest(test);
    String label[6]={"ALERT:Animals might rush","ALERT:Narrow lane","ALERT:Steep turns ahead","ALERT:speedbreaker Ahead","Danger-Alert","ALERT:Vehicles might skid"};
    String dir ="/users/srikanthreddy/desktop/final_project/trail";
            for(int k=0;k<dir.size();k++)
            {
            vector<String> objectInputFile;
            vector<String> sceneInputFile;
            String scene_link="/users/srikanthreddy/desktop/test1/trail/dataset"+to_string(k)+"/boards.jpg";
            String obj_link="/users/srikanthreddy/desktop/test1/trail/dataset"+to_string(k)+"/*.jpg";
            glob(obj_link,objectInputFile,false);
            glob(scene_link,sceneInputFile,false);
            for(int j=0;j<sceneInputFile.size();j++)
                {
                for(int i=0;i<objectInputFile.size();i++)
                    {
                    String sceneInputFile1=("/users/srikanthreddy/desktop/test1/trail/dataset"+to_string(k)+"/boards.jpg");//image in which we find
                    String objectInputFile1=("/users/srikanthreddy/desktop/test1/trail/dataset"+to_string(k)+"/a"+to_string(i)+".jpg"); //image to be matched
                    a.TMatches(sceneInputFile1,objectInputFile1,label[i]);
                     }
                }
            }
            return 0;
}
