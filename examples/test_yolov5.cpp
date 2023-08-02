#include "dl_algorithm.h"

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <memory>

using namespace std;


int main(int argc, char ** argv)
{

    if(argc < 3)
    {
        std::cout<<"input error,please input: ./yjh_deeplearning_test_yolov5 <yolov5_alg_cfg_path> <image_address>"<<std::endl;
        exit(0);
    }
    // string alg_cfg="/dongbangfa/work_code/DLAlg/config/yolov5.json";
    // string img_path = "/dongbangfa/data/wentie/u2data/yolov5_20220119_o_square/tmp/wentie_100_1.jpg";


    std::unique_ptr<yjh_deeplearning::AIAlgorithm> yolov5;
    yolov5.reset(new yjh_deeplearning::AIAlgorithm());
    int ret;
    ret = yolov5->Init(argv[1],0);
    

    if(ret != 0)
    {
        std::cout<<"Init error "<<ret<<std::endl;
        exit(0);
    }
    
    // 单张图片推理
    yjh_deeplearning::AIInputInfo input_info;
    yjh_deeplearning::AIOutputInfo output_info;

    cv::Mat test_image = cv::imread(argv[2]);
    if ( test_image.data == nullptr )
    {
        std::cout<<"read image error"<<std::endl;
        exit(0);
    }
    input_info.src_mat.emplace_back(std::make_shared<cv::Mat>(test_image));
    
    ret = yolov5->Inference(input_info,output_info);

    if(ret != 0)
    {
        std::cout<<"process error"<<std::endl;
        exit(0);
    }
    /*
        yolov5是目标检测任务，有多个目标
    */
   std::cout<<"single img result:"<<std::endl;
   for(unsigned int i=0;i<output_info.result_list.size();i++)
   {
        std::cout<<output_info.result_list[i].value<<" "<<output_info.result_list[i].score<<" "<<output_info.result_list[i].center_x<<" "<<output_info.result_list[i].center_y<<" "<<output_info.result_list[i].width<<" "<<output_info.result_list[i].height<<std::endl;
        
        cv::rectangle(test_image,cv::Rect(output_info.result_list[i].center_x-output_info.result_list[i].width/2.0,output_info.result_list[i].center_y-output_info.result_list[i].height/2.0,output_info.result_list[i].width,output_info.result_list[i].height),cv::Scalar(0,0,255),1,1,0);
        cv::putText(test_image,output_info.result_list[i].value,cv::Point(output_info.result_list[i].center_x,output_info.result_list[i].center_y),cv::FONT_HERSHEY_COMPLEX,2,cv::Scalar(0, 255, 255));
   } 
    cv::imwrite("test_result.jpg",test_image);
     //批量推理
    // std::vector<yjh_deeplearning::AIInputInfo> input_list;
    // std::vector<yjh_deeplearning::AIOutputInfo> output_list;
    // input_list.emplace_back(input_info);
    
    // if(argc > 3)
    // {
    //     cv::Mat test_image1 = cv::imread(argv[3]);
    //     if ( test_image1.data == nullptr )
    //     {
    //         std::cout<<"read image error"<<std::endl;
    //         exit(0);
    //     }
    //     input_info.src_mat.clear();
    //     input_info.src_mat.emplace_back(std::make_shared<cv::Mat>(test_image1));
    // }
    // input_list.emplace_back(input_info);
    // auto beforeTime = std::chrono::steady_clock::now();
    // for(int i=0;i<50;i++)
    // ret = yolov5->Inference(input_list,output_list);
    // auto afterTime = std::chrono::steady_clock::now();
    // double duration_millsecond = std::chrono::duration<double, std::milli>(afterTime - beforeTime).count();
    // std::cout << duration_millsecond/50 << "毫秒" << std::endl;

    // if(ret != 0)
    // {
    //     std::cout<<"process error"<<std::endl;
    //     exit(0);
    // }
    // std::cout<<"two batch imgs result:"<<std::endl;
    // for(unsigned int i=0;i<output_list.size();i++)
    // {
    //     std::cout<<i<<" result:"<<std::endl;
    //     for(unsigned int j=0;j<output_list[i].result_list.size();j++)
    //     {
    //         std::cout<<output_list[i].result_list[j].value<<" "<<output_list[i].result_list[j].score<<" "<<output_list[i].result_list[j].center_x<<" "<<output_list[i].result_list[j].center_y<<" "<<output_list[i].result_list[j].width<<" "<<output_list[i].result_list[j].height<<std::endl;
    //     } 
    // }


    return 0;
}
