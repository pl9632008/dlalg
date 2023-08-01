#include "dl_algorithm.h"

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
#define PI (3.1415926)

void DrawRotatedRect(cv::Mat& imgInoutput, cv::RotatedRect rectInput, cv::Scalar color, int thickness = 1, int lineType = cv::LINE_8);
 
void DrawRotatedRect(cv::Mat& imgInoutput, cv::RotatedRect rectInput, cv::Scalar color, int thickness, int lineType)
{
	cv::Point2f* vertices = new cv::Point2f[4];
	rectInput.points(vertices);
	for (int j = 0; j < 4; j++)
		line(imgInoutput, vertices[j], vertices[(j + 1) % 4], color, thickness, lineType);
}

int main(int argc, char ** argv)
{

    if(argc != 3)
    {
        std::cout<<"input error,please input: ./yjh_deeplearning_test_dbnet <dbnet_alg_cfg_path> <image_address>"<<std::endl;
        exit(0);
    }
    // string alg_cfg="/dongbangfa/work_code/DLAlg/config/oriened_rcnn.json";
    // string img_path = "/dongbangfa/data/wentie/u2data/yolov5_20220119_o_square/tmp/wentie_100_1.jpg";
    yjh_deeplearning::AIAlgorithm dbnet;


    int ret;
    ret = dbnet.Init(argv[1],0);
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
    //  ret = dbnet.Inference(input_info,output_info);
    auto beforeTime = std::chrono::steady_clock::now();
        // for(int i=0;i<10;i++)
    ret = dbnet.Inference(input_info,output_info);
    auto afterTime = std::chrono::steady_clock::now();
    double duration_millsecond = std::chrono::duration<double, std::milli>(afterTime - beforeTime).count();
    std::cout << duration_millsecond/10 << "毫秒" << std::endl;

    if(ret != 0)
    {
        std::cout<<"process error"<<std::endl;
        exit(0);
    }
    /*
        dbnet 是文字框目标检测任务，有多个目标
    */
   cv::RotatedRect rectInput;
   

   for(unsigned int i=0;i<output_info.result_list.size();i++)
   {
        std::cout<<output_info.result_list[i].value<<" "<<output_info.result_list[i].score<<" "<<output_info.result_list[i].center_x<<" "<<output_info.result_list[i].center_y<<" "<<output_info.result_list[i].width<<" "<<output_info.result_list[i].height<<" "<<output_info.result_list[i].angle<<std::endl;
         rectInput.center = cv::Point2f(output_info.result_list[i].center_x, output_info.result_list[i].center_y);
        rectInput.size = cv::Size(output_info.result_list[i].width, output_info.result_list[i].height);
        rectInput.angle = output_info.result_list[i].angle;
     
       
        DrawRotatedRect(test_image, rectInput, cv::Scalar(0, 0, 255));
        
    
       
   }    


    cv::imwrite("test_result.jpg",test_image);


    return 0;
}