#include "meter_recognition.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;



int main(int argc, char ** argv)
{

    if(argc != 4)
    {
        std::cout<<"input error,please input: ./test_recognition_ep <recognitiont_alg_cfg_path>  <defect_config>  <image_address1>"<<std::endl;
        exit(0);
    }
    // string alg_cfg="/dongbangfa/work_code/DLAlg/config/wentie.json";
    // string img_path = "/dongbangfa/data/wentie/u2data/wentie_20220119_o_square/tmp/wentie_100_1.jpg";
   

    int ret;
    void* alg_instance;
   
    ret = MeterLoadModel(argv[1],0,alg_instance);    
 
    if(ret != 0)
    {
        std::cout<<"Init error "<<ret<<std::endl;
        exit(0);
    }
    
    
    cv::Mat ori_img = cv::imread(argv[3]);
   

    
    std::vector<DETECT_RESULT> detect_result;
    ret = MeterRecognition(alg_instance,&ori_img,nullptr,argv[2],detect_result);
   if(ret != 0)
    {
        std::cout<<"process error"<<std::endl;
        exit(0);
    }


    for(unsigned int i=0;i<detect_result.size();i++)
   {
        std::cout<<detect_result[i].id<<std::endl;
        for(unsigned int j=0;j<detect_result[i].points.size();j++)
        {
            std::cout<<detect_result[i].points[j].x<<" "<<detect_result[i].points[j].y<<" | ";
            cv::circle(ori_img, cv::Point(detect_result[i].points[j].x,detect_result[i].points[j].y), 3, cv::Scalar(0, 255, 0),3);
        }     
        std::cout<<std::endl;
   } 
    cv::imwrite("test_result.jpg",ori_img);

    MeterReleaseModel(alg_instance);
}