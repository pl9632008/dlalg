#include "dl_algorithm.h"

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;


int main(int argc, char ** argv)
{

    if(argc != 3)
    {
        std::cout<<"input error,please input: ./yjh_deeplearning_test_u2net <u2net_alg_cfg_path> <image_address>"<<std::endl;
        exit(0);
    }
    // string alg_cfg="/dongbangfa/work_code/DLAlg/config/wentie.json";
    // string img_path = "/dongbangfa/data/wentie/u2data/wentie_20220119_o_square/tmp/wentie_100_1.jpg";
    yjh_deeplearning::AIAlgorithm wentie;


    int ret;
    ret = wentie.Init(argv[1],0);
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
    
    ret = wentie.Inference(input_info,output_info);

    if(ret != 0)
    {
        std::cout<<"process error"<<std::endl;
        exit(0);
    }
    /*
        u2net是分割任务，会返回分割后的二值图像，因为只有一类，所以output_info.reslut_list_个数是一个，第0号的seg_img_存储就是分割后的图像
    */

    std::shared_ptr<cv::Mat> result = std::static_pointer_cast<cv::Mat>(output_info.result_list[0].dst_mat);

    cv::imwrite("result.png",*result);


}