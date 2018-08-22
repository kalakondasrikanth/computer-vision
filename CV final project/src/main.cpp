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
    int c;
    TemplateMatching a;
    SVMtesttrain b;
    String test=("/users/srikanthreddy/desktop/CV/trail2/dataset1/boards.jpg"); //test image for SVM

    cout << "Please enter your choice:\n 1.Simple SVM to test Traffic signal recognition \n 2.Template Matching \n";
    cin >> c;
            switch(c)
            {
            case 1:b.trainntest(test);
                break;
            case 2:
                    String label0[6]={"ALERT:Animals around","ALERT:Narrow lane","ALERT:Steep turns ahead","ALERT:speedbreaker Ahead","ALERT:Danger","ALERT:Vehicles might skid"};
                    String label5[13]={"ALERT:speed limit 60KMPH","ALERT:speed limit 10KMPH","ALERT:speed limit 20KMPH","ALERT:speed limit 30KMPH","ALERT:speed limit 40KMPH","ALERT:speed limit 50KMPH","ALERT:Free from 50KMPH limit","ALERT:speed limit 70KMPH","ALERT:speed limit 50KMPH","ALERT:speed limit 80KMPH","ALERT:speed limit 90KMPH","ALERT:DONOT GO AT 50KMPH","ALERT:speed limit 100KMPH"};
                    String label1[1]={"ALERT:Speed limit is 100kmph"};
                    String label2[6]={"ALERT:Danger","ALERT:Speed limit 30KMPH","ALERT:Road might skid","ALERT:Road is inclined","ALERT:Steep curves ahead","ALERT:Danger"};
                    String label3[2]={"ALERT:Speed limit 70KMPH","ALERT:Speed breaker ahead"};
                    String label4[1]={"ALERT:100kmph"};

                String dir ="/users/srikanthreddy/desktop/CV/trail";
                        for(int k=0;k<dir.size();k++)
                        {
                        vector<String> objectInputFile;
                        vector<String> sceneInputFile;
                        String scene_link="/users/srikanthreddy/desktop/CV/trail/dataset"+to_string(k)+"/boards.jpg";
                        String obj_link="/users/srikanthreddy/desktop/CV/trail/dataset"+to_string(k)+"/*.jpg";
                        glob(obj_link,objectInputFile,false);
                        glob(scene_link,sceneInputFile,false);
                        for(int j=0;j<sceneInputFile.size();j++)
                            {
                            for(int i=0;i<objectInputFile.size();i++)
                                {
                                String sceneInputFile1=("/users/srikanthreddy/desktop/CV/trail/dataset"+to_string(k)+"/boards.jpg");//image in which we find
                                String objectInputFile1=("/users/srikanthreddy/desktop/CV/trail/dataset"+to_string(k)+"/a"+to_string(i)+".jpg"); //image to be matched
                                switch(k)
                                                     {
                                                    case 0:
                                                        a.TMatches(sceneInputFile1,objectInputFile1,label0[i]);
                                                        break;
                                                     case 1:
                                                        a.TMatches(sceneInputFile1,objectInputFile1,label1[i]);
                                                        break;
                                                      case 2:
                                                        a.TMatches(sceneInputFile1,objectInputFile1,label2[i]);
                                                        break;
                                                      case 3:
                                                        a.TMatches(sceneInputFile1,objectInputFile1,label3[i]);
                                                        break;
                                                      case 4:
                                                        a.TMatches(sceneInputFile1,objectInputFile1,label4[i]);
                                                        break;
                                                    case 5:
                                                      a.TMatches(sceneInputFile1,objectInputFile1,label5[i]);
                                                      break;
                                                      }
                                 }
                            }
                        }
            }
            return 0;
}
