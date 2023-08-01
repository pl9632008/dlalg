#include "dl_dbnet.h"
#include "dl_common.h"

#include <glog/logging.h>

namespace yjh_deeplearning{



int DbnetAlg::Init(dlalg_jsons::AlgInfo &algInfo)
{
       
    if(algInfo.model_list.size() == 0)
    {
        LOG(ERROR) << "DbnetModel  can not set model" ;
        return YJH_AI_ALG_INIT_ERROR;
    }
 
  
    dbnet_model_ = std::make_shared<DbnetModel>(algInfo.model_list[0]);
    return dbnet_model_->InitModel();
 }


int DbnetAlg::ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)
{
  
    if(input_info.src_mat.size() ==0 || input_info.src_mat[0] == nullptr)
    {
        return YJH_AI_INPUT_IMG_ERROR;
    }
    std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_info.src_mat[0]);    
    CHECK_SUCCESS(dbnet_model_->ModelPredict(*ori_img,detection_boxes_));
    output_info.result_list.clear();
    AIResult result; 
    cv::RotatedRect rectInput;   
    cv::Point2f pts[4];
    for(unsigned int i=0;i<detection_boxes_.size();i++)
    {   
        rectInput = cv::minAreaRect(detection_boxes_[i].box_point_);
        result.score = detection_boxes_[i].score_;
        result.center_x = rectInput.center.x;
        result.center_y = rectInput.center.y;
        result.width = rectInput.size.width;
        result.height = rectInput.size.height;
        result.angle = rectInput.angle;
        output_info.result_list.emplace_back(result);
    }
    return DLSUCCESSED;
}

int DbnetAlg::DeInit()
{
    if(dbnet_model_ != nullptr)
    {
        dbnet_model_->DeInitModel();
    }
    
    return DLSUCCESSED;
}


REGISTERALG(dbnet_dlalg, DbnetAlg);


}