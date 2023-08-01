#include "dl_common.h"
#include "dl_algorithm.h"
#include "defect_detector_c_api.h"

#include <string>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace yjh_deeplearning{

yjh_deeplearning::PatrolSystemDefectRecognition *GetSBAlgorithmInstance()
{
    if(!google::IsGoogleLoggingInitialized())
	{
		FLAGS_log_dir = "./logs/";
		FLAGS_logbufsecs = 0;
		FLAGS_max_log_size = 2;
		google::InitGoogleLogging("YJHDLALG");
	}
    LOG(INFO) << "get patrol system SB defect recognition instance!";
    return new yjh_deeplearning::PatrolSystemDefectRecognition();
}

void DestorySBAlgorithmInstance(yjh_deeplearning::PatrolSystemDefectRecognition *instance)
{

    if(instance != nullptr)
    {
        delete instance;
        instance = nullptr;
    }
    LOG(INFO) << "destory patrol system SB defect recognition instance!";
}

int InitSBAlgorithm(yjh_deeplearning::PatrolSystemDefectRecognition *self, char *alg_conf_file, int gpu_index)
{
    LOG(INFO) << "init patrol system SB defect:"<< alg_conf_file <<", gpu_index:  "<< gpu_index;
    if(self != nullptr)
    {
        return self->Init(alg_conf_file, gpu_index);
    }
    else
    {
        return yjh_deeplearning::DLFAILED;
    }
}

int ProcessSBImage(yjh_deeplearning::PatrolSystemDefectRecognition *self, char **img_path_list, int input_size, DefectOutput **output_infos)
{
    LOG(INFO) << "patrol system SB defect processing images ...";
    if(self != nullptr && img_path_list != nullptr && output_infos != nullptr)
    {
        std::vector<std::string> imgpathlist(input_size);
        for(int i=0; i<input_size; i++)
        {
            imgpathlist[i].assign(img_path_list[i]);
            LOG(INFO) << "imgpathlist[" << i <<"] " << imgpathlist[i];
        }

        std::vector<yjh_deeplearning::AIOutputInfo> output_list;
        std::vector<std::string> img_info;
        int ret = self->ProcessImage(imgpathlist, output_list, img_info);
        if(ret != yjh_deeplearning::DLSUCCESSED)
        {
            LOG(ERROR) << "process image failed";
            return ret;
        }
        if(img_info.size() != input_size)
        {
            LOG(ERROR) << "process error, img flag size is error";
            return ret;
        }

        int c_err = 0;
        for(unsigned int i=0; i<img_info.size();i++)
        {
            if(img_info[i] == "ImgError")
            {
                output_infos[i]->result_size = 0;
                std::strcpy(output_infos[i]->img_flag, img_info[i].data());
                c_err += 1;
                LOG(INFO) << "'ImgError' output_infos[" << i << "]->img_flag: " << output_infos[i]->img_flag;
                continue;
            }
             
            int c_cor = i - c_err;
            if(output_list[c_cor].result_list.size() == 0)
            {
                output_infos[i]->result_size = 0;
                std::strcpy(output_infos[i]->img_flag, img_info[i].data());
                LOG(INFO) << "'No res' output_infos[" << i << "]->img_flag: " << output_infos[i]->img_flag;
                continue;
            }

            DefectResult *single_res = new DefectResult[output_list[c_cor].result_list.size()];
            output_infos[i]->result_list = single_res;
            output_infos[i]->result_size = output_list[c_cor].result_list.size();
            std::strcpy(output_infos[i]->img_flag, img_info[i].data());
            LOG(INFO) << "output_infos[" << i << "]->img_flag: " << output_infos[i]->img_flag;
 

            for(unsigned int j=0;j<output_list[c_cor].result_list.size();j++) 
            {
                LOG(INFO) << "c++ output_list[" << i << "][" << j << "], class: " << output_list[c_cor].result_list[j].value
                << ", score_: " << output_list[c_cor].result_list[j].score << ", center_x: " << output_list[c_cor].result_list[j].center_x << ", center_y: "
                << ", width_: " << output_list[c_cor].result_list[j].width << ", height: " << output_list[c_cor].result_list[j].height;
                std::strcpy(single_res[j].class_name, output_list[c_cor].result_list[j].value.data());          
                single_res[j].score    = output_list[c_cor].result_list[j].score;
                single_res[j].center_x = output_list[c_cor].result_list[j].center_x;
                single_res[j].center_y = output_list[c_cor].result_list[j].center_y;
                single_res[j].width    = output_list[c_cor].result_list[j].width;
                single_res[j].height   = output_list[c_cor].result_list[j].height;
                LOG(INFO) << "c single_res[" << i << "][" << j << "], class_name: " << single_res[j].class_name
                << ", score_: " << single_res[j].score << ", center_x: " << single_res[j].center_x << ", center_y: "
                << ", width_: " << single_res[j].width << ", height: " << single_res[j].height;
            }
        }

        LOG(INFO) << "completing  images process";
        return yjh_deeplearning::DLSUCCESSED;
    }
    else
    {
        LOG(ERROR) << "ProcessSBImage nullptr ";
        return yjh_deeplearning::DLFAILED;
    }
}


int ProcessSBImageJson(yjh_deeplearning::PatrolSystemDefectRecognition *self, char **img_path_list, int input_size,const char** output_json)
{
    // std::cout << std::hex << (int *)output_json << std::endl;
    LOG(INFO) << "patrol system SB defect processing images ...";
    if(self != nullptr && img_path_list != nullptr && output_json != nullptr)
    {
        std::vector<std::string> imgpathlist(input_size);
        for(int i=0; i<input_size; i++)
        {
            imgpathlist[i].assign(img_path_list[i]);
            LOG(INFO) << "imgpathlist[" << i <<"] " << imgpathlist[i];
        }

        static std::string output_infos;
        int ret = self->ProcessImageJson(imgpathlist, output_infos);
        if(ret != yjh_deeplearning::DLSUCCESSED)
        {
            LOG(ERROR) << "process image failed";
            return ret;
        }

        *output_json = output_infos.c_str();

        LOG(INFO) << "output_json: " << *output_json;
        LOG(INFO) << "completing  images process";
        return yjh_deeplearning::DLSUCCESSED;
    }
    else
    {
        LOG(ERROR) << "ProcessSBImage nullptr ";
        return yjh_deeplearning::DLFAILED;
    }
}


}


