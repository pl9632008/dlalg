#include "dl_algorithm.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <memory>

#include "nlohmann/json.hpp"
#include <fstream>

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

   std::cout<<"single img result:"<<std::endl;
   for(unsigned int i=0;i<output_info.result_list.size();i++)
   {
        std::cout<<output_info.result_list[i].value<<" "<<output_info.result_list[i].score<<" "<<output_info.result_list[i].center_x<<" "<<output_info.result_list[i].center_y<<" "<<output_info.result_list[i].width<<" "<<output_info.result_list[i].height<<std::endl;
        
        cv::rectangle(test_image,cv::Rect(output_info.result_list[i].center_x-output_info.result_list[i].width/2.0,output_info.result_list[i].center_y-output_info.result_list[i].height/2.0,output_info.result_list[i].width,output_info.result_list[i].height),cv::Scalar(0,0,255),1,1,0);
        cv::putText(test_image,output_info.result_list[i].value,cv::Point(output_info.result_list[i].center_x,output_info.result_list[i].center_y),cv::FONT_HERSHEY_COMPLEX,2,cv::Scalar(0, 255, 255));
   } 
    cv::imwrite("../test_result.jpg",test_image);

    nlohmann::json jsonObj;
    for(size_t i =0 ; i<output_info.result_list.size();i++){
            nlohmann::json jsonTmp;
            jsonTmp["value"]=output_info.result_list[i].value;
            jsonTmp["score"]=output_info.result_list[i].score;
            jsonTmp["center_x"]=output_info.result_list[i].center_x;
            jsonTmp["center_y"]=output_info.result_list[i].center_y;
            jsonTmp["width"]=output_info.result_list[i].width;
            jsonTmp["height"]=output_info.result_list[i].height;
            jsonObj.emplace_back(jsonTmp);
    }
    std::string res = jsonObj.dump();
    std::ofstream outfile("../output.json");
    outfile<<res;

    return 0;
}
