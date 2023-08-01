#include "face_recognition.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

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
    
    cv::Mat test_image = cv::imread(argv[2]);
    if ( test_image.data == nullptr )
    {
        std::cout<<"read image error"<<std::endl;
        exit(0);
    }
    
    
    
    float *features1 = new float[512];
    int size1=512;
    std::vector<uchar> vecImg;
	std::vector<int> vecCompression_params;
	vecCompression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
	vecCompression_params.push_back(90);	
	cv::imencode(".jpg", test_image, vecImg, vecCompression_params);
    std::cout<<"vecImg1 size: "<<vecImg.size()<<std::endl;
    ret = ProcessFaceImage(facerecognition,(char*)vecImg.data(),vecImg.size()-1000,features1,&size1);

    


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

     cv::Mat test_image2 = cv::imread(argv[3]);

    
    if ( test_image2.data == nullptr )
    {
        std::cout<<"read image error"<<std::endl;
        exit(0);
    }
    
         float *features2 = new float[512];
          int size2=512;

    std::cout<<(int)test_image2.at<char>(0,0)<<" "<<(int)test_image2.at<char>(0,1)<<std::endl;  

     vecImg.clear();

	cv::imencode(".jpg", test_image2, vecImg, vecCompression_params);
    std::cout<<"vecImg2 size: "<<vecImg.size()<<std::endl;
     ret = ProcessFaceImage(facerecognition,(char*)vecImg.data(),vecImg.size()-9000,features2,&size2);
    

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