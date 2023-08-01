#include "dl_alg_keypoint.h"
#include "dl_common.h"
#include "dl_util.h"
#include <glog/logging.h>


namespace yjh_deeplearning{


int KeypointDLAlg::Init(dlalg_jsons::AlgInfo &algInfo)
{
    if(algInfo.model_list.size()!=2 )
    {
        return YJH_AI_ALG_INIT_ERROR;
    }

    if(algInfo.model_list[1].preprocess_list.size() != 0)
    {
        keypoint_height_ = algInfo.model_list[1].preprocess_list[0].img_height;
        keypoint_width_ = algInfo.model_list[1].preprocess_list[0].img_width;
    }
    else
    {
        LOG(ERROR)<<"keypoint model input size not config";
        return DLFAILED;
    }   
   
    detection_model_ = std::make_shared<Yolov5Model>(algInfo.model_list[0]);
    keypoint_model_ = std::make_shared<LiteHrnetModel>(algInfo.model_list[1]);
    class_name_.swap(algInfo.model_list[0].class_name);
    
    
    CHECK_SUCCESS(detection_model_->InitModel());
    CHECK_SUCCESS(keypoint_model_->InitModel());
    return DLSUCCESSED;
}


int KeypointDLAlg::ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)
{    
    
    if(input_info.src_mat.size() ==0 || input_info.src_mat[0] == nullptr )
    {
        return YJH_AI_INPUT_IMG_ERROR;
    }
    std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_info.src_mat[0]);    
    CHECK_SUCCESS(detection_model_->ModelPredict(*ori_img,detection_results_));
    output_info.result_list.clear();
    AIResult result;   
    std::vector<cv::Point> pointVec; 
  
    for(unsigned int i=0;i<detection_results_.size();i++)
    {
        if(detection_results_[i].class_idx<class_name_.size())
        {            
            result.value = class_name_[detection_results_[i].class_idx];
            result.score = detection_results_[i].score;
            
            result.center_x = detection_results_[i].center_x;
            result.center_y = detection_results_[i].center_y;
            result.width = detection_results_[i].width;
            result.height = detection_results_[i].height;
            cv::Mat rot_imt = GetRoiExpandImg(*ori_img,detection_results_[i],keypoint_height_,keypoint_width_,expanding_,inverse_warp_mat_);
           
            keypoint_model_->ModelPredict(rot_imt,pointVec);
            result.key_points.clear();
            cv::Point p;
            for(unsigned int i=0;i<pointVec.size();i++)
            {                
                p.x = inverse_warp_mat_.ptr<double>(0)[0] * pointVec[i].x + inverse_warp_mat_.ptr<double>(0)[1] * pointVec[i].y + inverse_warp_mat_.ptr<double>(0)[2];
                p.y = inverse_warp_mat_.ptr<double>(1)[0] * pointVec[i].x + inverse_warp_mat_.ptr<double>(1)[1] * pointVec[i].y + inverse_warp_mat_.ptr<double>(1)[2];
                result.key_points.emplace_back(std::make_pair(p.x,p.y));
            }
            output_info.result_list.emplace_back(result);          
        }
        else
        {
            LOG(ERROR) << "class idx greater than class name size";
        }
    }
    return DLSUCCESSED;
    
}

int KeypointDLAlg::DeInit()
{   
    if(detection_model_ != nullptr)
    {
        detection_model_->DeInitModel();
    }
    if(keypoint_model_ != nullptr)
    {
        keypoint_model_->DeInitModel();
    }
    return DLSUCCESSED;
}

REGISTERALG(keypoint_dlalg, KeypointDLAlg);


}