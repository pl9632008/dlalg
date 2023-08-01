#include "face_recognition.h"
#include <iostream>
#include <string>
// #include <opencv2/opencv.hpp>

using namespace std;



int main(int argc, char ** argv)
{

    if(argc != 4)
    {
        std::cout<<"input error,please input: ./yjh_deeplearning_test_face_alg <facerecognitiont_alg_cfg_path> <image_address1> <image_address2>"<<std::endl;
        exit(0);
    }
    // string alg_cfg="/dongbangfa/work_code/DLAlg/config/wentie.json";
    // string img_path = "/dongbangfa/data/wentie/u2data/wentie_20220119_o_square/tmp/wentie_100_1.jpg";
    yjh_deeplearning::FaceRecognitionAlgorithm *facerecognition;


    int ret;
    facerecognition = GetFaceRecognitionAlgorithmInstance();
    
    ret = InitFaceRecognitionAlgorithm(facerecognition,argv[1]);
    if(ret != 0)
    {
        std::cout<<"Init error "<<ret<<std::endl;
        exit(0);
    }
    
    
    // cv::Mat ori_img = cv::imread(argv[2]);
    float *features1 = new float[512];
    int size1=512;
    int center_x,center_y,width,height;
   
    ret = ProcessFaceLocalImage(facerecognition,argv[2],&center_x,&center_y,&width,&height,features1,&size1);

    //  cv::rectangle(ori_img,cv::Rect(center_x-width/2.0,center_y-height/2.0,width,height),cv::Scalar(0,0,255),1,1,0);
    // cv::imwrite("1_ori_img.jpg",ori_img);


    if(ret != 0)
    {
        std::cout<<"process error"<<std::endl;
        exit(0);
    }
     for(int i=0;i<512;i++)
   {
    std::cout<<features1[i]<<" ";
    
   }
    std::cout<<std::endl;
    std::cout<<"size1: "<<size1<<std::endl;

    //  ori_img = cv::imread(argv[3]);

    
    // if ( ori_img.data == nullptr )
    // {
    //     std::cout<<"read image error"<<std::endl;
    //     exit(0);
    // }
    
         float *features2 = new float[512];
          int size2=512;


    
     ret = ProcessFaceLocalImage(facerecognition,argv[3],&center_x,&center_y,&width,&height,features1,&size1);

    //   cv::rectangle(ori_img,cv::Rect(center_x-width/2.0,center_y-height/2.0,width,height),cv::Scalar(0,0,255),1,1,0);
    // cv::imwrite("2_ori_img.jpg",ori_img);

std::cout<<"size2: "<<size2<<std::endl;
   

    if(ret != 0)
    {
        std::cout<<"process error"<<std::endl;
        exit(0);
    }
    /*
        人脸识别任务，返回mat是一个一维数组
    */
   
   std::cout<<"similarity result: "<<IsSamePeople(facerecognition,features1,size1,features2,size2)<<std::endl;
    // cv::imwrite("result.png",*result);

    DestoryFaceRecognitionAlgorithmInstance(facerecognition);
}