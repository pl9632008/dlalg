#include "dl_algorithm.h"

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
#define PI (3.1415926)



int main(int argc, char ** argv)
{

    if(argc != 3)
    {
        std::cout<<"input error,please input: ./yjh_deeplearning_test_crnn <crnn_alg_cfg_path> <image_address>"<<std::endl;
        exit(0);
    }
    // string alg_cfg="/dongbangfa/work_code/DLAlg/config/oriened_rcnn.json";
    // string img_path = "/dongbangfa/data/wentie/u2data/yolov5_20220119_o_square/tmp/wentie_100_1.jpg";
    yjh_deeplearning::AIAlgorithm crnn;


    int ret;
    ret = crnn.Init(argv[1],0);
    if(ret != 0)
    {
        std::cout<<"Init error "<<ret<<std::endl;
        exit(0);
    }

    yjh_deeplearning::AIInputInfo input_info;
    yjh_deeplearning::AIOutputInfo output_info;

    cv::Mat test_image = cv::imread(argv[2]);
   
    if ( test_image.data == nullptr )
    {
        std::cout<<"read image error"<<std::endl;
        exit(0);
    }
    input_info.src_mat.emplace_back(std::make_shared<cv::Mat>(test_image));
     ret = crnn.Inference(input_info,output_info);
    auto beforeTime = std::chrono::steady_clock::now();
        for(int i=0;i<10;i++)
    ret = crnn.Inference(input_info,output_info);
    auto afterTime = std::chrono::steady_clock::now();
    double duration_millsecond = std::chrono::duration<double, std::milli>(afterTime - beforeTime).count();
    std::cout << duration_millsecond/10 << "毫秒" << std::endl;

    if(ret != 0)
    {
        std::cout<<"process error"<<std::endl;
        exit(0);
    }
    /*
        crnn 是文字识别任务，只有一个目标
    */
 

  
   std::cout<<output_info.result_list[0].value<<std::endl;
  


    return 0;
}