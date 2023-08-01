#include "face_recognition.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;



int main(int argc, char ** argv)
{

    if(argc != 4)
    {
        std::cout<<"input error,please input: ./yjh_deeplearning_test_facecpp_alg <facerecognitiont_alg_cfg_path> <image_address1> <image_address2>"<<std::endl;
        exit(0);
    }
    // string alg_cfg="/dongbangfa/work_code/DLAlg/config/wentie.json";
    // string img_path = "/dongbangfa/data/wentie/u2data/wentie_20220119_o_square/tmp/wentie_100_1.jpg";
    yjh_deeplearning::FaceRecognitionAlgorithm facerecognition;


    int ret;
    
    
    ret = facerecognition.Init(argv[1]);
    if(ret != 0)
    {
        std::cout<<"Init error "<<ret<<std::endl;
        exit(0);
    }
    
    cv::Mat test_image = cv::imread(argv[2]);
    if ( test_image.data == nullptr )
    {
        std::cout<<"read image error"<<std::endl;
        exit(0);
    }
    
  
    int size1;

    std::vector<float> features1;
    ret = facerecognition.ProcessImage(std::make_shared<cv::Mat>(test_image),features1);

    if(ret != 0)
    {
        std::cout<<"process error"<<std::endl;
        exit(0);
    }
     for(int i=0;i<10;i++)
   {
    std::cout<<features1[i]<<" ";
    
   }
    std::cout<<std::endl;
    std::cout<<"size1: "<<size1<<std::endl;

     cv::Mat test_image2 = cv::imread(argv[3]);

    
    if ( test_image2.data == nullptr )
    {
        std::cout<<"read image error"<<std::endl;
        exit(0);
    }
    
        
          int size2;

    std::cout<<(int)test_image2.at<char>(0,0)<<" "<<(int)test_image2.at<char>(0,1)<<std::endl;  

    std::vector<float> features2;
    ret = facerecognition.ProcessImage(std::make_shared<cv::Mat>(test_image2),features2);
    
      for(int i=0;i<10;i++)
   {
    std::cout<<features2[i]<<" ";
    
   }
std::cout<<"size2: "<<size2<<std::endl;
   

    if(ret != 0)
    {
        std::cout<<"process error"<<std::endl;
        exit(0);
    }
    /*
        人脸识别任务，返回mat是一个一维数组
    */
   
   

   std::cout<<"similarity result: "<<facerecognition.IsSamePeople(features1,features2)<<std::endl;
    // cv::imwrite("result.png",*result);


}