#include "defect_detector_c_api.h"
#include <iostream>
#include <string>
#include "dirent.h"
#include <sys/stat.h>
#include <opencv2/opencv.hpp>

using namespace std;


int main(int argc, char ** argv)
{

    if(argc != 2)
    {
        std::cout<<"[test_patrol_system_c] input error,please input: ./test_patrol_system <recognitiont_alg_cfg_path>"<<std::endl;
        exit(0);
    }
    // string alg_cfg="/dongbangfa/work_code/DLAlg/config/wentie.json";
    // string img_path = "/dongbangfa/data/wentie/u2data/wentie_20220119_o_square/tmp/wentie_100_1.jpg";
    yjh_deeplearning::PatrolSystemDefectRecognition* alg_instance = yjh_deeplearning::GetSBAlgorithmInstance();

    int ret;
    ret = yjh_deeplearning::InitSBAlgorithm(alg_instance, argv[1], 0);
    if(ret != 0)
    {
        std::cout<<"[test_patrol_system_c] init error "<< ret << std::endl;
        exit(0);
    }
    
    // std::vector<std::string> img_path_list{"/ai_server/test_data/five_img/bj_bpmh.jpg", "/ai_server/test_data/five_img/abc.jpg",
    //                                        "/ai_server/test_data/five_img/wcgz.jpg", "/ai_server/test_data/five_img/data_err.jpg"};
    // std::vector<std::string> img_path_list{"/ai_server/test_data/five_img/bj_bpmh.jpg", "/ai_server/test_data/five_img/abc.jpg",
    //                                        "/ai_server/test_data/five_img/wcgz.jpg", "/ai_server/test_data/five_img/data_err.jpg",
    //                                        "/ai_server/test_data/five_img/hzyw.jpg", "/ai_server/test_data/five_img/yw_nc.jpg"};
    std::vector<std::string> img_path_list{"/ai_server/test_data/five_img/wcgz.jpg"};

    // 开辟内存空间, 将多条字符串赋值给char **, 方式1
    // char* c_path_list[img_path_list.size()];
    // for(unsigned i=0;i<img_path_list.size();i++)
    // {
    //     c_path_list[i] =  new char[img_path_list[i].length() + 1];
    //     std::strcpy(c_path_list[i], img_path_list[i].c_str());
    // }

    // 开辟内存空间, 将多条字符串赋值给char **, 方式2
    char** c_path_list = new char*[img_path_list.size()];
    for (std::size_t i = 0; i < img_path_list.size(); i++) {
        c_path_list[i] = strdup(img_path_list.at(i).c_str());
    }

    for (std::size_t i = 0; i < img_path_list.size(); i++) {
        std::cout <<"[test_patrol_system_c] input c_path[" << i <<"]: " <<  c_path_list[i] << std::endl;
    } 

    // 创建保存结果的文件夹
    std::string save_dir = "./res_c_yolov5";
    DIR* filepath = NULL;
    if((filepath = opendir(save_dir.c_str())) == NULL)
    {
        int ret = mkdir(save_dir.c_str(), 0775);
        if(ret != 0)
        {
            std::cout << "[test_patrol_system_c] failed to create folder '" << save_dir << "'" << std::endl;
            return -1;
        }
        std::cout << "[test_patrol_system_c] created save path '" << save_dir << "'" << std::endl;
    }
    else
    {
        std::cout << "[test_patrol_system_c] '" << save_dir << "' exists" << std::endl;
    }

    #if 1
    std::cout << "*******************推理结果以结构体格式返回****************" << std::endl;
    // 给多张图片检测结果开辟内存空间
    // yjh_deeplearning::DefectOutput *detect_results = new yjh_deeplearning::DefectOutput[img_path_list.size()]; // 错误示例
    // ret = yjh_deeplearning::ProcessSBImage(alg_instance, c_path_list, img_path_list.size(), &detect_results);  // 错误示例
    yjh_deeplearning::DefectOutput** detect_results = new yjh_deeplearning::DefectOutput*[img_path_list.size()];
    for(unsigned int i=0; i<img_path_list.size(); i++) 
    {
        detect_results[i] = new yjh_deeplearning::DefectOutput[1];
    }

    // 将多张图片送入推理，将多张图片结果保存在detect_results中
    ret = yjh_deeplearning::ProcessSBImage(alg_instance, c_path_list, img_path_list.size(), detect_results);
    if(ret != 0)
    {
        std::cout << "[test_patrol_system_c] process error" << std::endl;
        exit(0);
    }

    // 解析结果，并将结果画在图片上
    for(unsigned int i=0; i<img_path_list.size(); i++)
    {
        std::cout<<"[test_patrol_system_c] " << img_path_list[i] << ", img_flag: " << detect_results[i]->img_flag << std::endl;
        if(std::string(detect_results[i]->img_flag) == "ImgError")
        {
            std::cout<<"[test_patrol_system_c] read image error: " << img_path_list[i] << std::endl;
            continue;
        }
        
        cv::Mat cv_img = cv::imread(img_path_list[i]);

        // 获取图片名
        std::string img_name = img_path_list[i].substr(img_path_list[i].find_last_of("/") + 1, -1);
        std::cout <<  "[test_patrol_system_c] c img_name: " << img_name << std::endl;
        std::string img_save_path = save_dir + "/" + "struct_" + img_name;

        std::cout << "[test_patrol_system_c] '" << img_path_list[i] << "' results: " << std::endl;
        for(unsigned int j=0; j<detect_results[i]->result_size; j++)
        {
            std::cout << "\t\t\t" << detect_results[i]->result_list[j].class_name << " " << detect_results[i]->result_list[j].score << " "
            << detect_results[i]->result_list[j].center_x << " " << detect_results[i]->result_list[j].center_y << " "
            << detect_results[i]->result_list[j].width << " " << detect_results[i]->result_list[j].height << std::endl;
            
            cv::rectangle(cv_img, cv::Rect(detect_results[i]->result_list[j].center_x - detect_results[i]->result_list[j].width / 2.0,
                                        detect_results[i]->result_list[j].center_y - detect_results[i]->result_list[j].height / 2.0,
                                        detect_results[i]->result_list[j].width, detect_results[i]->result_list[j].height),
                                        cv::Scalar(0, 0, 255), 1, 1, 0);
            cv::putText(cv_img, detect_results[i]->result_list[j].class_name, 
                                cv::Point(detect_results[i]->result_list[j].center_x, detect_results[i]->result_list[j].center_y),
                                cv::FONT_HERSHEY_COMPLEX,2, cv::Scalar(0, 255, 255));
        }

        std::cout << "[test_patrol_system_c] img_save_path: " << img_save_path << "\n" << std::endl;
        imwrite(img_save_path, cv_img);
    }
    // 释放开辟的内存空间
    for (std::size_t i = 0; i<img_path_list.size(); i++) {
        delete[] detect_results[i]->result_list; 
    }
    delete[] detect_results;
    #endif

    #if 1
    // 将多张图片送入推理，将多张图片结果保存在json_results中
    const char *json_results;

    std::cout << "*******************推理结果以json string格式返回****************" << std::endl;
    ret = yjh_deeplearning::ProcessSBImageJson(alg_instance, c_path_list, img_path_list.size(), &json_results);
    if(ret != 0)
    {  
        std::cout << "[test_patrol_system_c] process error" << std::endl;
        exit(0);
    }
    std::cout << "jsonresults_out: " << json_results << std::endl;
    // std::cout << "out: " << std::hex << (int *)json_results << std::endl;
    #endif

    // 释放开辟的内存空间
    for (std::size_t i = 0; i<img_path_list.size(); i++) {
        free(c_path_list[i]);
    }
    delete[] c_path_list;

    yjh_deeplearning::DestorySBAlgorithmInstance(alg_instance);

    return 0;
}