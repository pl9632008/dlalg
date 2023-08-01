#include "dl_alg_tensorrt.h"
#include "dl_common.h"
#include "dl_tensorrt.h"
#include <glog/logging.h>


namespace yjh_deeplearning{


int TRTDLAlg::Init(dlalg_jsons::AlgInfo &algInfo)
{
    if(algInfo.model_list.size()!=1 )
    {
        return YJH_AI_ALG_INIT_ERROR;
    }
    class_name_.swap(algInfo.model_list[0].class_name);
 
    trt_model_detect_normal_ = std::make_shared<TenorrtPredictor>(algInfo.model_list[0]);
 
    CHECK_SUCCESS(trt_model_detect_normal_->InitModel());
    
    return DLSUCCESSED;
}


int TRTDLAlg::ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)
{    
    
    output_info.result_list.clear();
    if(input_info.src_mat.size()==0 || input_info.src_mat[0] == nullptr)
    {
        return YJH_AI_INPUT_IMG_ERROR;
    }
    std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_info.src_mat[0]);
    cv::Mat image = ori_img->clone();
   
  
    // cv::Mat image;
    // cv::merge(std::vector<cv::Mat>{imageClone,imageClone,imageClone}, image);
    
    
    CHECK_SUCCESS(trt_model_detect_normal_->GeneralPreProcess(image));
    CHECK_SUCCESS(trt_model_detect_normal_->BatchPredict({{image},{image}},out_tensors_));
    for(int i=0;i<out_tensors_[0].size[0];i++)
    {
        for(int j=0;j<out_tensors_[0].size[1];j++) 
        {
            LOG(ERROR)<<out_tensors_[0].at<float>(i,j);
          
        }
    }
 
    LOG(ERROR)<<out_tensors_[0].dims<<" "<<out_tensors_[0].type();
    for(int i=0;i<out_tensors_[0].dims;i++)
    {
        LOG(ERROR)<<out_tensors_[0].size[i];
    }

    cv::minMaxLoc(out_tensors_[0], NULL, &classProb_, NULL, &classNumber_);
    
    AIResult tmpResult;
  
    tmpResult.value =  class_name_[int(classNumber_.x)];   
    
    tmpResult.score = classProb_;
    output_info.result_list.emplace_back(tmpResult);
    return DLSUCCESSED;
    
}

int TRTDLAlg::DeInit()
{
    trt_model_detect_normal_->DeInitModel();
    return DLSUCCESSED;
}

REGISTERALG(trt_dlalg, TRTDLAlg);


}