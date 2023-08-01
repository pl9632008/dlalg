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
        std::cout<<"input error,please input: ./yjh_deeplearning_test_keypoint <kepoint_alg_cfg_path> <image_address>"<<std::endl;
        exit(0);
    }
   
    yjh_deeplearning::AIAlgorithm keypoint;


    int ret;
    ret = keypoint.Init(argv[1],0);
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
   
    ret = keypoint.Inference(input_info,output_info);

    if(ret != 0)
    {
        std::cout<<"process error"<<std::endl;
        exit(0);
    }
    /*
        关键点检测是检测一组坐标
    */
   for(unsigned int i=0;i<output_info.result_list.size();i++)
   {
        std::cout<<output_info.result_list[i].value<<" "<<output_info.result_list[i].score<<" "<<output_info.result_list[i].center_x<<" "<<output_info.result_list[i].center_y<<" "<<output_info.result_list[i].width<<" "<<output_info.result_list[i].height<<std::endl;
        
        cv::rectangle(test_image,cv::Rect(output_info.result_list[i].center_x-output_info.result_list[i].width/2.0,output_info.result_list[i].center_y-output_info.result_list[i].height/2.0,output_info.result_list[i].width,output_info.result_list[i].height),cv::Scalar(0,0,255),1,1,0);
        cv::putText(test_image,output_info.result_list[i].value,cv::Point(output_info.result_list[i].center_x,output_info.result_list[i].center_y),cv::FONT_HERSHEY_COMPLEX,2,cv::Scalar(0, 255, 255));
   } 
   
    // static const int joint_pairs[12][2] = {
	// {0, 1}, {1, 3}, {3, 5}, {0, 2}, {2, 4}, {4, 6}, {1, 7}, {7, 9}, {2, 8}, {8, 10},{1,2},{7,8}};

        static const int joint_pairs[12][2] = {
	{0, 1}};
   for(unsigned int i=0;i<output_info.result_list.size();i++)
   {    
       std::vector<cv::Point> keyPoints;
		

        for(unsigned int j =0 ;j<output_info.result_list[i].key_points.size();j++)
        {
            keyPoints.push_back(cv::Point(output_info.result_list[i].key_points[j].first,output_info.result_list[i].key_points[j].second));
            cv::circle(test_image, cv::Point(output_info.result_list[i].key_points[j].first,output_info.result_list[i].key_points[j].second), 3, cv::Scalar(0, 255, 0),3);           
        }
        for (unsigned int j =0 ;j<1;j++)
		{
			cv::line(test_image, keyPoints[joint_pairs[j][0]], keyPoints[joint_pairs[j][1]], cv::Scalar(255, 0, 255),2);
		}


      
   }
    cv::imwrite("test_result.jpg",test_image);

}