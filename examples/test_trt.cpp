#include "dl_algorithm.h"
#include<fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;


int main(int argc, char ** argv)
{

    if(argc != 3)
    {
        std::cout<<"input error,please input: ./yjh_deeplearning_test_jufang <jufang_alg_cfg_path> <jufang_data_path>"<<std::endl;
        exit(0);
    }
    string alg_cfg="/dongbangfa/work_code/DLAlg/config/jufang.json";
    string txt_path = "/dongbangfa/study/out_11.txt";
    yjh_deeplearning::AIAlgorithm jufang;


    int ret;
    ret = jufang.Init(argv[1],0);
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
   
    ret = jufang.Inference(input_info,output_info);

    if(ret != 0)
    {
        std::cout<<"process error"<<std::endl;
        exit(0);
    }
    /*
        局放检测是单分类任务，会返回分类结果，因为只有一类，所以output_info.reslut_list_个数是一个，第0号的class_存储就是预测的类名，score_是类置信度
    */

    std::string class_name = output_info.result_list[0].value;
    float score = output_info.result_list[0].score;
    std::cout<<"class_name: "<<class_name<<" score： "<<score<<std::endl;

}