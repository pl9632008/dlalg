#include "dl_alg_cls.h"
#include "dl_common.h"
#include "dl_opencvdnn.h"
#include <glog/logging.h>


namespace yjh_deeplearning{


int ClsDLAlg::Init(dlalg_jsons::AlgInfo &algInfo)
{
    if(algInfo.model_list.size()!=1 )
    {
        return YJH_AI_ALG_INIT_ERROR;
    }
    if(algInfo.model_list[0].auto_preprocess.size() != 0)
    {
        auto_preprocess_ = true;
    }   
    class_name_.swap(algInfo.model_list[0].class_name);
  
    cls_model_ = std::make_shared<OpencvDNNPredictor>(algInfo.model_list[0]);   
    CHECK_SUCCESS(cls_model_->InitModel());  
    return DLSUCCESSED;
}


int ClsDLAlg::ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)
{    
    
    output_info.result_list.clear();
    if(input_info.src_mat.size()==0 || input_info.src_mat[0] == nullptr)
    {
        return YJH_AI_INPUT_IMG_ERROR;
    }
    std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_info.src_mat[0]);
    cv::Mat image = ori_img->clone();
    if(auto_preprocess_ == false)
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);   
        CHECK_SUCCESS(cls_model_->GeneralPreProcess(image));
    }    
    CHECK_SUCCESS(cls_model_->Predict(image,out_tensors_));
    CHECK_SUCCESS(cls_model_->Softmax(out_tensors_[0]));
      
    cv::minMaxLoc(out_tensors_[0], NULL, &classProb_, NULL, &classNumber_);
    
    AIResult tmpResult;  
    
    tmpResult.value =  class_name_[int(classNumber_.x)]; 
    tmpResult.score = classProb_;
    output_info.result_list.emplace_back(tmpResult);
    return DLSUCCESSED;
    
}



REGISTERALG(cls_dlalg, ClsDLAlg);


}