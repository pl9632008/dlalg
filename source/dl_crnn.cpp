#include "dl_crnn.h"
#include "dl_common.h"

#include <glog/logging.h>

namespace yjh_deeplearning{



int CRNNtAlg::Init(dlalg_jsons::AlgInfo &algInfo)
{
       
    if(algInfo.model_list.size() == 0)
    {
        LOG(ERROR) << "DbnetModel  can not set model" ;
        return YJH_AI_ALG_INIT_ERROR;
    }
 
  
    crnn_model_ = std::make_shared<CRNNModel>(algInfo.model_list[0]);
    return crnn_model_->InitModel();
 }


int CRNNtAlg::ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)
{
  
    if(input_info.src_mat.size() ==0 || input_info.src_mat[0] == nullptr)
    {
        return YJH_AI_INPUT_IMG_ERROR;
    }
    std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_info.src_mat[0]); 
    std::string voc_str; 
    CHECK_SUCCESS(crnn_model_->ModelPredict(*ori_img,voc_str));
    output_info.result_list.clear();
    AIResult result; 
    result.value = voc_str;
    output_info.result_list.emplace_back(result);
    return DLSUCCESSED;
}

int CRNNtAlg::DeInit()
{
    if( crnn_model_ != nullptr)
    {
        crnn_model_->DeInitModel();
    }
    
    return DLSUCCESSED;
}


REGISTERALG(crnn_dlalg, CRNNtAlg);


}