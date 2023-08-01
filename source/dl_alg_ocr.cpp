#include "dl_alg_ocr.h"
#include "dl_common.h"
#include "dl_util.h"
#include "dl_onnxruntime.h"
#include <glog/logging.h>


namespace yjh_deeplearning{


int OCRDLAlg::Init(dlalg_jsons::AlgInfo &algInfo)
{
    if(algInfo.model_list.size()!=2 )
    {
        return YJH_AI_ALG_INIT_ERROR;
    }
    
    // if(algInfo.model_list[1].preprocess_list.size() != 0)
    // {
    //     model1_height_ = algInfo.model_list[1].preprocess_list[0].img_height;
    //     model1_width_ = algInfo.model_list[1].preprocess_list[0].img_width;
    // }
    // else
    // {
    //     LOG(ERROR)<<"keypoint model input size not config";
    //     return DLFAILED;
    // } 
    // if(algInfo.model_list[1].auto_preprocess.size() != 0)
    // {
    //     auto_preprocess1_ = true;
    // }   
    if(algInfo.model_list[0].model_name == "dbnet")
    {
        det_model_ = std::make_shared<DbnetModel>(algInfo.model_list[0]);
    }
    else
    {
        LOG(ERROR)<<"not config ocr det model";
        return YJH_AI_ALG_INIT_ERROR;
    }

    if(algInfo.model_list[1].model_name == "crnn")
    {
        rec_model_ = std::make_shared<CRNNModel>(algInfo.model_list[1]);
    }
    else
    {
        LOG(ERROR)<<"not config ocr rec model";
        return YJH_AI_ALG_INIT_ERROR;
    }

    class_name_.swap(algInfo.model_list[1].class_name);
        
    // angle_model_ = std::make_shared<ONNXRuntimePredictor>(algInfo.model_list[1]); 
    
    CHECK_SUCCESS(det_model_->InitModel());
    CHECK_SUCCESS(rec_model_->InitModel());
    // CHECK_SUCCESS(angle_model_->InitModel());

    return DLSUCCESSED;
}


int OCRDLAlg::ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)
{    
    output_info.result_list.clear();
    
    if(input_info.src_mat.size() ==0 || input_info.src_mat[0] == nullptr )
    {       
        return YJH_AI_INPUT_IMG_ERROR;
    }
    if(det_model_ == nullptr || rec_model_ == nullptr )
    {
         LOG(ERROR)<<"not config ocr det or rec model";
        return YJH_AI_NETWORK_LOAD_ERROR;
    }
    std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_info.src_mat[0]);   
    CHECK_SUCCESS(det_model_->ModelPredict(*ori_img,det_boxes_));

    cv::Mat roi_img,angel_mat;
    std::string ocr_str;
    AIResult result; 
    cv::RotatedRect rectInput;   
    cv::Point2f pts[4];
    for(unsigned int i=0;i<det_boxes_.size();i++)
    {
        roi_img = GetRotateCropImage(*ori_img,det_boxes_[i].box_point_);
        // cv::imwrite(std::to_string(i)+"_aaa.jpg",roi_img);
        // angel_mat = roi_img.clone();
        // CHECK_SUCCESS(AdjustTargetImg(angel_mat,model1_width_,model1_height_));
        // if(auto_preprocess1_ == false)
        // {
        //     cv::cvtColor(image, image, cv::COLOR_BGR2RGB);   
        //     CHECK_SUCCESS(angle_model_->GeneralPreProcess(image));
        // }  
        // CHECK_SUCCESS(cls_model_->Predict(image,out_tensors_));
        // CHECK_SUCCESS(cls_model_->Softmax(out_tensors_[0]));
        
        // cv::minMaxLoc(out_tensors_[0], NULL, &classProb_, NULL, &classNumber_);
        rec_model_->ModelPredict(roi_img,ocr_str);
 
        if(class_name_.size() != 0)
        {
            auto iter = std::find(std::begin(class_name_), std::end(class_name_), ocr_str);
            if(iter == class_name_.end())
            {
                continue;
            }
        }

        rectInput = cv::minAreaRect(det_boxes_[i].box_point_);
        result.value = ocr_str;
        result.score = det_boxes_[i].score_;
        result.center_x = rectInput.center.x;
        result.center_y = rectInput.center.y;
        result.width = rectInput.size.width;
        result.height = rectInput.size.height;
        result.angle = rectInput.angle;
        output_info.result_list.emplace_back(result);
    }
    return DLSUCCESSED;
}

int OCRDLAlg::DeInit()
{   
    if(det_model_ != nullptr)
    {
        det_model_->DeInitModel();
    }
    if(rec_model_ != nullptr)
    {
        rec_model_->DeInitModel();
    }
    return DLSUCCESSED;
}

REGISTERALG(ocr_dlalg, OCRDLAlg);

}