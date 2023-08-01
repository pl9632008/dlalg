#include "dl_alg_u2net.h"
#include "dl_common.h"
#include "dl_opencvdnn.h"
#include <glog/logging.h>


namespace yjh_deeplearning{


int U2NetDLAlg::Init(dlalg_jsons::AlgInfo &algInfo)
{
    if(algInfo.model_list.size()!=1 )
    {
        return YJH_AI_ALG_INIT_ERROR;
    }
    wentie_model_ = std::make_shared<OpencvDNNPredictor>(algInfo.model_list[0]);
    return wentie_model_->InitModel();    
}


int U2NetDLAlg::ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)
{        
    output_info.result_list.clear();
    if(input_info.src_mat.size() ==0 || input_info.src_mat[0]== nullptr)
    {
        return YJH_AI_INPUT_IMG_ERROR;
    }
    std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_info.src_mat[0]);
    cv::Mat image = ori_img->clone();
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    CHECK_SUCCESS(wentie_model_->PreProcessByMulMaxValue(image));
    CHECK_SUCCESS(wentie_model_->Predict(image,out_tensors_));
  
    const int rows = out_tensors_[0].size[2];
    const int cols = out_tensors_[0].size[3];
    cv::Mat maxVal(rows, cols, CV_32FC1, out_tensors_[0].data);
    
    cv::resize(maxVal, o_img_, cv::Size(ori_img->cols,ori_img->rows),0,0, cv::INTER_LINEAR);   
    o_img_.convertTo(o_img_, CV_8U,255);
    
    AIResult tmpResult;    
    tmpResult.dst_mat = std::make_shared<cv::Mat>(o_img_);
    output_info.result_list.emplace_back(tmpResult);
    return DLSUCCESSED;
    
}

int U2NetDLAlg::DeInit()
{   
    if(wentie_model_ != nullptr)
    {
        wentie_model_->DeInitModel();
    }
    return DLSUCCESSED;
}



REGISTERALG(u2net_dlalg, U2NetDLAlg);


}