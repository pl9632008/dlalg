#include "defect_detector.h"
#include <iostream>
#include <string>
#include "dirent.h"
#include <sys/stat.h>
#include "nlohmann/json.hpp"
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char ** argv)
{

    if(argc != 2)
    {
        std::cout<<"input error,please input: ./yjh_deeplearning_test_patrol_system <facerecognitiont_alg_cfg_path>"<<std::endl;
        exit(0);
    }
    // string alg_cfg="/dongbangfa/work_code/DLAlg/config/wentie.json";
    // string img_path = "/dongbangfa/data/wentie/u2data/wentie_20220119_o_square/tmp/wentie_100_1.jpg";
    yjh_deeplearning::PatrolSystemDefectRecognition patrol_system_sb;

    int ret;
    
    ret = patrol_system_sb.Init(argv[1], 0);
    if(ret != 0)
    {
        std::cout<<"Init error "<<ret<<std::endl;
        exit(0);
    }
    
    // std::vector<std::string> img_path_list{"/ai_server/test_data/five_img/bj_bpmh.jpg", "/ai_server/test_data/five_img/abc.jpg",
    //                                        "/ai_server/test_data/five_img/wcgz.jpg", "/ai_server/test_data/five_img/data_err.jpg"};
    // std::vector<std::string> img_path_list{"/ai_server/test_data/five_img/bj_bpmh.jpg", "/ai_server/test_data/five_img/abc.jpg",
    //                                        "/ai_server/test_data/five_img/wcgz.jpg", "/ai_server/test_data/five_img/data_err.jpg",
    //                                        "/ai_server/test_data/five_img/hzyw.jpg", "/ai_server/test_data/five_img/yw_nc.jpg"};
    std::vector<std::string> img_path_list{"/ai_server/test_data/five_img/wcgz.jpg"};

    // 创建保存结果的文件夹
    std::string save_dir = "./res_yolov5";
    DIR* filepath = NULL;
    if((filepath = opendir(save_dir.c_str())) == NULL)
    {
        int ret = mkdir(save_dir.c_str(), 0775);
        if(ret != 0)
        {
            std::cout << "CheckTestPortPlugRecognition: failed to create folder '" << save_dir << "'" << std::endl;
            return -1;
        }
        std::cout << "CheckTestPortPlugRecognition: created save path '" << save_dir << "'" << std::endl;
    }
    else
    {
        std::cout << "CheckTestPortPlugRecognition: '" << save_dir << "' exists" << std::endl;
    }

    #if 1
    // *******************推理结果以定义的结构返回****************//
    std::cout << "*******************推理结果以结构体格式返回****************" << std::endl;
    std::vector<yjh_deeplearning::AIOutputInfo> output_list;
    std::vector<std::string> img_info;
    ret = patrol_system_sb.ProcessImage(img_path_list, output_list, img_info);

    if(ret != 0)
    {
        std::cout<<"process error"<<std::endl;
        exit(0);
    }
     
    // 解析output_list结果，并将结果画在图片上
    int c_err = 0;
    if(img_info.size() == img_path_list.size()) 
    {
        for(unsigned int i=0; i<img_path_list.size(); i++)
        {
            // 读取图片
            if(img_info[i] == "ImgError")
            {
                std::cout<<"read image error: " << img_path_list[i]<< std::endl;
                c_err = c_err + 1;
                continue;
            }
            cv::Mat cv_img = cv::imread(img_path_list[i]);
            
            // 获取图片名
            std::string img_name = img_path_list[i].substr(img_path_list[i].find_last_of("/") + 1, -1);
            std::cout <<  "img_name: " << img_name << std::endl;
            std::string img_save_path = save_dir + "/" + "struct_" + img_name;

            // 解析结果，画图
            int c_cor = i - c_err;
            for(unsigned int j=0; j<output_list[c_cor].result_list.size(); j++)
            {
                std::cout << output_list[c_cor].result_list[j].value << " " << output_list[c_cor].result_list[j].score << " "
                << output_list[c_cor].result_list[j].center_x << " " << output_list[c_cor].result_list[j].center_y << " "
                << output_list[c_cor].result_list[j].width << " " << output_list[c_cor].result_list[j].height << std::endl;
                
                cv::rectangle(cv_img, cv::Rect(output_list[c_cor].result_list[j].center_x - output_list[c_cor].result_list[j].width  / 2.0,
                                            output_list[c_cor].result_list[j].center_y - output_list[c_cor].result_list[j].height / 2.0,
                                            output_list[c_cor].result_list[j].width, output_list[c_cor].result_list[j].height),
                                            cv::Scalar(0, 0, 255), 1, 1, 0);
                cv::putText(cv_img, output_list[c_cor].result_list[j].value, 
                                    cv::Point(output_list[c_cor].result_list[j].center_x, output_list[c_cor].result_list[j].center_y),
                                    cv::FONT_HERSHEY_COMPLEX,2, cv::Scalar(0, 255, 255));
            }

            std::cout << "img_save_path: " << img_save_path  << std::endl;
            imwrite(img_save_path, cv_img);
        }
    }
    else
    {
        std::cout << "img_path_list size: " << img_path_list.size() << ", output_list size: " << output_list.size() << std::endl;
    }
    #endif

    #if 1
    // *******************推理结果以json string格式返回****************//
    std::cout << "*******************推理结果以json string格式返回****************" << std::endl;
    std::string json_output;
    ret = patrol_system_sb.ProcessImageJson(img_path_list, json_output);
    if(ret != 0)
    {
        std::cout<<"process error"<<std::endl;
        exit(0);
    }

    nlohmann::json json_list = nlohmann::json::parse(json_output);
    std::cout << json_list.dump() << std::endl;
    // 解析json_list结果，并将结果画在图上
    if(json_list.size() == img_path_list.size()) 
    {

        for(unsigned int i=0; i < img_path_list.size(); i++)
        {
            // 获取图片名
            std::string img_name = img_path_list[i].substr(img_path_list[i].find_last_of("/") + 1, -1);
            // std::cout <<  "img_name: " << img_name << std::endl;
            std::string img_save_path = save_dir + "/" + "json_" + img_name;

            cv::Mat cv_img = cv::imread(img_path_list[i]);
            if(cv_img.data == nullptr)
            {
                std::cout<<"read image error: " << img_path_list[i]<< std::endl;
                continue;
            }
            
            nlohmann::json detect_res = json_list[img_path_list[i]];
            if (detect_res["code"] == "OK")
            {
                for(unsigned j=0; j < detect_res["objects"].size(); j++)
                {   
                    nlohmann::json obj = detect_res["objects"][j];

                    cv::rectangle(cv_img, cv::Rect(obj["box"][0].get<int>() - obj["box"][2].get<int>() / 2.0, obj["box"][1].get<int>() - obj["box"][3].get<int>() / 2.0, obj["box"][2], obj["box"][3]),
                                  cv::Scalar(0, 0, 255), 1, 1, 0);
                    // cv::rectangle(cv_img, cv::Rect(obj["box"][0] - obj["box"][2] / 2.0, obj["box"][1] - obj["box"][3] / 2.0, obj["box"][2], obj["box"][3]),
                    //               cv::Scalar(0, 0, 255), 1, 1, 0);
                    cv::putText(cv_img, obj["class"], cv::Point(obj["box"][0], obj["box"][1]),
                                cv::FONT_HERSHEY_COMPLEX,2, cv::Scalar(0, 255, 255));
                }
            }
            imwrite(img_save_path, cv_img);
            std::cout << img_path_list[i] << " results: " << detect_res.dump() << std::endl;   
        }
    }
    else
    {
        std::cout << "img_path_list size: " << img_path_list.size() << ", output_list size: " << json_list.size() << std::endl;
    }
    #endif

    return 0;


}