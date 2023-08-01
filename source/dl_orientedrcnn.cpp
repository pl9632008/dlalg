#include "dl_orientedrcnn.h"
#include "dl_common.h"

#include <glog/logging.h>

namespace yjh_deeplearning{



int OrientedRcnnAlg::Init(dlalg_jsons::AlgInfo &algInfo)
{
    if(algInfo.model_list.size() == 0 || algInfo.model_list[0].class_name.size() <= 0)
    {
        LOG(ERROR) << "OrientedRcnnAlg  model parameter error" ;
        return YJH_AI_ALG_INIT_ERROR;
    }
    class_name_.swap(algInfo.model_list[0].class_name);
  
    orientedrcnn_model_ = std::make_shared<OrientedRcnnModel>(algInfo.model_list[0]);
    return orientedrcnn_model_->InitModel();
 }


int OrientedRcnnAlg::ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)
{

    output_info.result_list.clear();
    if(input_info.src_mat.size() ==0 || input_info.src_mat[0] == nullptr)
    {
        LOG(ERROR) << "OrientedRcnnAlg inference input error" ;
        return YJH_AI_INPUT_IMG_ERROR;
    }
    std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_info.src_mat[0]);    
    CHECK_SUCCESS(orientedrcnn_model_->ModelPredict(*ori_img,detection_results));  
    AIResult result;    
    for(unsigned int i=0;i<detection_results.size();i++)
    {
        if(detection_results[i].class_idx<class_name_.size())
        {
            result.value = class_name_[detection_results[i].class_idx];
            result.score = detection_results[i].score;
            result.angle = detection_results[i].angle;
            result.center_x = detection_results[i].center_x;
            result.center_y = detection_results[i].center_y;
            result.width = detection_results[i].width;
            result.height = detection_results[i].height;
            output_info.result_list.emplace_back(result);
        }
        else
        {
            LOG(ERROR) << "class idx greater than class name size";
        }
    }
    
    return DLSUCCESSED;
}

int OrientedRcnnAlg::DeInit()
{
    if(orientedrcnn_model_ != nullptr){
        orientedrcnn_model_->DeInitModel();
    }
    
    return DLSUCCESSED;
}

REGISTERALG(orientedrcnn_dlalg, OrientedRcnnAlg);


}