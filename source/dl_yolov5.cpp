#include "dl_yolov5.h"
#include "dl_common.h"

#include <glog/logging.h>

namespace yjh_deeplearning{



int Yolov5Alg::Init(dlalg_jsons::AlgInfo &algInfo)
{
       
    if(algInfo.model_list.size() == 0 || algInfo.model_list[0].class_name.size() <= 0)
    {
        LOG(ERROR) << "yolo v5 can not set class" ;
        return YJH_AI_ALG_INIT_ERROR;
    }
    class_name_.swap(algInfo.model_list[0].class_name);
    class_thresh_.swap(algInfo.model_list[0].class_thresh);
  
    yolov5_model_ = std::make_shared<Yolov5Model>(algInfo.model_list[0]);
    return yolov5_model_->InitModel();
 }


int Yolov5Alg::ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)
{
  
    if(input_info.src_mat.size() ==0 || input_info.src_mat[0] == nullptr)
    {
        return YJH_AI_INPUT_IMG_ERROR;
    }
    std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_info.src_mat[0]);    
    CHECK_SUCCESS(yolov5_model_->ModelPredict(*ori_img,detection_results_));
    output_info.result_list.clear();
    AIResult result;    
    for(unsigned int i=0;i<detection_results_.size();i++)
    {
        if(detection_results_[i].class_idx<class_name_.size())
        {            
            result.value = class_name_[detection_results_[i].class_idx];
            result.score = detection_results_[i].score;
            if(class_thresh_.find(result.value) != class_thresh_.end() && result.score < class_thresh_.find(result.value)->second)
            {
                continue;
            }
            result.center_x = detection_results_[i].center_x;
            result.center_y = detection_results_[i].center_y;
            result.width = detection_results_[i].width;
            result.height = detection_results_[i].height;
            output_info.result_list.emplace_back(result);          
        }
        else
        {
            LOG(ERROR) << "class idx greater than class name size";
        }
    }
    return DLSUCCESSED;
}

int Yolov5Alg::ProcessPic(const std::vector<AIInputInfo> &input_list,std::vector<AIOutputInfo> &output_list)
{   
    std::vector<cv::Mat> imgs;
    for(unsigned int i=0;i<input_list.size();i++)
    {
        if(input_list[i].src_mat.size() ==0 || input_list[i].src_mat[0] == nullptr)
        {
            return YJH_AI_INPUT_IMG_ERROR;
        }
        std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_list[i].src_mat[0]);       
        imgs.emplace_back(*ori_img);
    }  
    CHECK_SUCCESS(yolov5_model_->ModelPredict(imgs,total_detection_results_));
    output_list.clear();
    // LOG(ERROR) << "message: "<<total_detection_results_.size() ;
    AIResult result;
    AIOutputInfo result_output;
    for(unsigned i = 0;i<total_detection_results_.size();i++)
    {   
        result_output.result_list.clear();
        for(unsigned int j=0;j<total_detection_results_[i].size();j++)
        {
            if(total_detection_results_[i][j].class_idx<class_name_.size())
            {
                result.value = class_name_[total_detection_results_[i][j].class_idx];
                result.score = total_detection_results_[i][j].score;
                if(class_thresh_.find(result.value) != class_thresh_.end() && result.score < class_thresh_.find(result.value)->second)
                {
                    continue;
                }
                result.center_x = total_detection_results_[i][j].center_x;
                result.center_y = total_detection_results_[i][j].center_y;
                result.width = total_detection_results_[i][j].width;
                result.height = total_detection_results_[i][j].height;
                result_output.result_list.emplace_back(result);
            }
            else
            {
                LOG(ERROR) << "class idx greater than class name size";
            }
        }       
        output_list.emplace_back(result_output);
    }

    return DLSUCCESSED;
}

int Yolov5Alg::DeInit()
{
    if(yolov5_model_ != nullptr)
    {
        yolov5_model_->DeInitModel();
    }
    
    return DLSUCCESSED;
}


REGISTERALG(yolov5_dlalg, Yolov5Alg);


}