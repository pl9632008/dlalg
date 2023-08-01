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
    for(unsigned int i = 0;i< detect_result.size();i++)
    {
        std::cout<<detect_result[i].id<<" "<<detect_result[i].points.size()<<std::endl;
    }

    int x,y,width,height; 
    for(unsigned int i=0;i<detect_result.size();i++)
   {
        if(detect_result[i].value == "good")
        {
            continue;
        }

        x=detect_result[i].points[0].x;
        y=detect_result[i].points[0].y;
        width = detect_result[i].points[1].x - detect_result[i].points[0].x;
        height =  detect_result[i].points[2].y - detect_result[i].points[1].y;

         std::cout<<x<<" "<<y<<" "<<width<<" "<<height<<" "<<std::endl;
        cv::rectangle(ori_img,cv::Rect(x,y,width,height),cv::Scalar(0,0,255),1,1,0);
        cv::putText(ori_img,std::to_string(detect_result[i].id),cv::Point(x,y),cv::FONT_HERSHEY_COMPLEX,2,cv::Scalar(0, 255, 255));
   } 
    cv::imwrite("test_result.jpg",ori_img);

    MeterReleaseModel(alg_instance);
}