#include "dl_alg_scrfd.h"
#include "dl_common.h"

#include <glog/logging.h>


namespace yjh_deeplearning{


int SCRFDDLAlg::Init(dlalg_jsons::AlgInfo &algInfo)
{
    if(algInfo.model_list.size()!=1)
    {
        return YJH_AI_ALG_INIT_ERROR;
    }
    scrfd_model_ = std::make_shared<SCRFDModel>(algInfo.model_list[0]);
    return scrfd_model_->InitModel();
   
}



int SCRFDDLAlg::ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)
{
  
    if(input_info.src_mat.size() ==0 || input_info.src_mat[0] == nullptr)
    {
        return YJH_AI_INPUT_IMG_ERROR;
    }
    std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_info.src_mat[0]);    
    CHECK_SUCCESS(scrfd_model_->ModelPredict(*ori_img,output_info));
    
    return DLSUCCESSED;
}

int SCRFDDLAlg::DeInit()
{
    if( scrfd_model_ != nullptr)
    {
        scrfd_model_->DeInitModel();
    }
    
    return DLSUCCESSED;
}
 

REGISTERALG(scrfd_dlalg, SCRFDDLAlg);


}