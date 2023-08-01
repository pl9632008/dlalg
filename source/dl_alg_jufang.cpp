#include "dl_alg_jufang.h"
#include "dl_common.h"
#include "dl_opencvdnn.h"
#include <glog/logging.h>


namespace yjh_deeplearning{


int JufangDLAlg::Init(dlalg_jsons::AlgInfo &algInfo)
{
    if(algInfo.model_list.size()!=2 )
    {
        return YJH_AI_ALG_INIT_ERROR;
    }
    class_name_normal_.swap(algInfo.model_list[0].class_name);
    class_name_fangdian_.swap(algInfo.model_list[1].class_name);
    jufang_model_detect_normal_ = std::make_shared<OpencvDNNPredictor>(algInfo.model_list[0]);
    jufang_model_detect_fangdian_ = std::make_shared<OpencvDNNPredictor>(algInfo.model_list[1]);
    CHECK_SUCCESS(jufang_model_detect_normal_->InitModel());
    CHECK_SUCCESS(jufang_model_detect_fangdian_->InitModel());
    return DLSUCCESSED;
}


int JufangDLAlg::ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)
{    
    
    output_info.result_list.clear();
    if(input_info.src_mat.size()==0 || input_info.src_mat[0] == nullptr)
    {
        return YJH_AI_INPUT_IMG_ERROR;
    }
    std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_info.src_mat[0]);
    cv::Mat imageClone = ori_img->clone();
    if(imageClone.type() != CV_32SC1)
    {
        LOG(ERROR)<<"image type  error "<<imageClone.type();
        return YJH_AI_INPUT_IMGTYPE_ERROR;
    }
  
    cv::Mat image;
    cv::merge(std::vector<cv::Mat>{imageClone,imageClone,imageClone}, image);
    
    
    CHECK_SUCCESS(jufang_model_detect_normal_->GeneralPreProcess(image));
    CHECK_SUCCESS(jufang_model_detect_normal_->Predict(image,out_tensors_));
    CHECK_SUCCESS(jufang_model_detect_normal_->Softmax(out_tensors_[0]));
   
    cv::minMaxLoc(out_tensors_[0], NULL, &classProb_, NULL, &classNumber_);
    
    AIResult tmpResult;
  
    if(int(classNumber_.x) == 0)
    {
        tmpResult.value =  class_name_normal_[int(classNumber_.x)];       
    }
    else
    {
        // CHECK_SUCCESS(jufang_model_detect_fangdian_->GeneralPreProcess(image));
        CHECK_SUCCESS(jufang_model_detect_fangdian_->Predict(image,out_tensors_));
        CHECK_SUCCESS(jufang_model_detect_fangdian_->Softmax(out_tensors_[0]));
        cv::minMaxLoc(out_tensors_[0], NULL, &classProb_, NULL, &classNumber_);      
        tmpResult.value =  class_name_fangdian_[int(classNumber_.x)];        
    }
    
    tmpResult.score = classProb_;
    output_info.result_list.emplace_back(tmpResult);
    return DLSUCCESSED;
    
}



REGISTERALG(jufang_dlalg, JufangDLAlg);


}