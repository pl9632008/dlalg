#include "dl_alg_face_recognition.h"
#include "dl_util.h"
#include "dl_common.h"
#include "dl_opencvdnn.h"

#include <glog/logging.h>


namespace yjh_deeplearning{


int FaceRecognitionDLAlg::Init(dlalg_jsons::AlgInfo &algInfo)
{
    if(algInfo.model_list.size()!=2 )
    {
        return YJH_AI_ALG_INIT_ERROR;
    }
    if(algInfo.model_list[0].model_name == "scrfd")
    {
        scrfd_model_ = std::make_shared<SCRFDModel>(algInfo.model_list[0]);
        CHECK_SUCCESS(scrfd_model_->InitModel());
    }
    else
    {    
        LOG(ERROR)<<"wrong detection model";
        return DLFAILED;
    }
    if(algInfo.model_list[1].model_name == "mobileface")
    {
         mobileface_model_ = std::make_shared<OpencvDNNPredictor>(algInfo.model_list[1]);   
        CHECK_SUCCESS(mobileface_model_->InitModel());  
    }
    else
    {
        LOG(ERROR)<<"wrong recognition model";
        return DLFAILED;
    }

    src_mat_align_ = cv::Mat(5,2,CV_32FC1);  
    memcpy(src_mat_align_.data, src_align, 2 * 5 * sizeof(float));

    dst_mat_align_ = cv::Mat(5,2,CV_32FC1);
   
    return DLSUCCESSED;
}


int FaceRecognitionDLAlg::ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)
{      
    if(input_info.src_mat.size() ==0 || input_info.src_mat[0] == nullptr)
    {
        return YJH_AI_INPUT_IMG_ERROR;
    }
    std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_info.src_mat[0]);   
    AIOutputInfo detect_output_info;  
    CHECK_SUCCESS(scrfd_model_->ModelPredict(*ori_img,detect_output_info));
    if(detect_output_info.result_list.size() ==0 )
    {
        LOG(ERROR)<<"no face detect";
        return YJH_AI_FACE_NONE_ERROR;
    }
    int index=0;
    if(detect_output_info.result_list.size() > 1 )
    {   
        int maxarea=0;
        for(unsigned int i=0;i<detect_output_info.result_list.size();i++)
        {
            if(detect_output_info.result_list[i].width*detect_output_info.result_list[i].height > maxarea)
            {
                index = i;
                maxarea = detect_output_info.result_list[i].width*detect_output_info.result_list[i].height;
            }
        }       
    }
    cv::Mat image = ori_img->clone();
    // LOG(ERROR)<<detect_output_info.result_list_[index].center_x_<<" "<<detect_output_info.result_list_[index].center_y_<<" "<<detect_output_info.result_list_[index].width_<<" "<<detect_output_info.result_list_[index].height_;
    for(unsigned int i=0;i<detect_output_info.result_list[index].key_points.size();i++)
    {
        dst_point_[i*2] = detect_output_info.result_list[index].key_points[i].first;
        dst_point_[i*2+1] = detect_output_info.result_list[index].key_points[i].second;
    }
    memcpy(dst_mat_align_.data, dst_point_, 2 * 5 * sizeof(float));
    cv::Mat m = similarTransform(dst_mat_align_ ,src_mat_align_);
   
    cv::Mat map_matrix= m(cv::Rect(0, 0, 3, 2));
    //align
    
    cv::warpAffine(image, face_roi_, map_matrix, { (int)STANDARDWIDTH, (int)STANDARDWIDTH});
    // cv::imwrite("image.jpg",image);
    // cv::imwrite("face.jpg",face_roi_);
    cv::cvtColor(face_roi_, face_roi_, cv::COLOR_BGR2RGB);     
    CHECK_SUCCESS(mobileface_model_->GeneralPreProcess(face_roi_));
    out_tensors_.clear();   
    CHECK_SUCCESS(mobileface_model_->Predict(face_roi_,out_tensors_));
   
    output_info.result_list.clear();
    AIResult tmpResult;
    tmpResult.center_x = detect_output_info.result_list[index].center_x;
    tmpResult.center_y = detect_output_info.result_list[index].center_y;
    tmpResult.width = detect_output_info.result_list[index].width;
    tmpResult.height = detect_output_info.result_list[index].height; 
    tmpResult.dst_mat = std::make_shared<cv::Mat>(out_tensors_[0]);
    output_info.result_list.emplace_back(tmpResult);
   
    return DLSUCCESSED;
    
}

int FaceRecognitionDLAlg::DeInit()
{   
    if(scrfd_model_ != nullptr)
    {
        scrfd_model_->DeInitModel();
    }
    if(mobileface_model_ != nullptr)
    {
        mobileface_model_->DeInitModel();
    }
    return DLSUCCESSED;
}

REGISTERALG(face_recognition_dlalg, FaceRecognitionDLAlg);


}