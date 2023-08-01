#include "dl_litehrnet_model.h"
#include "dl_common.h"
#include "dl_opencvdnn.h"
#include "dl_onnxruntime.h"
#include "dl_tensorrt.h"
#include <glog/logging.h>


namespace yjh_deeplearning{

LiteHrnetModel::LiteHrnetModel(dlalg_jsons::ModelInfo &modelInfo)
{
    preprocess_list_ = modelInfo.preprocess_list;  
    infer_engine_ = modelInfo.infer_engine;
    if(modelInfo.auto_preprocess.size() != 0)
    {
        auto_preprocess_ = true;
    } 
    if(modelInfo.stride.size()>0)
    {
        stride_ = modelInfo.stride[0];
    }
    if(infer_engine_ == INFER_ENGINE_ORT)
    {
        model_ = std::make_shared<ONNXRuntimePredictor>(modelInfo); 
    }
    else if(infer_engine_ == INFER_ENGINE_TRT)
    {
        model_ = std::make_shared<TenorrtPredictor>(modelInfo); 
    }
    else
    {
        model_ = nullptr;
    }
        
      
 }

int LiteHrnetModel::InitModel()
{

    if(preprocess_list_.size() == 0 || model_ == nullptr)
    {
        LOG(ERROR) << "preprocess config error or model is nullptr";
        return YJH_AI_PARAMETER_ERROR;
    }
    img_height_ = preprocess_list_[0].img_height;
    img_width_ = preprocess_list_[0].img_width;
    return model_->InitModel();
}


int LiteHrnetModel::ModelPredict(const cv::Mat &img, std::vector<cv::Point> &pointVec)
{      
    cv::Mat image = img.clone();  
    if(auto_preprocess_ == false)
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);   
        CHECK_SUCCESS(model_->GeneralPreProcess(image));
    } 
    
    CHECK_SUCCESS(model_->Predict(image,out_tensors_));      
    CHECK_SUCCESS(PostProcess(out_tensors_[0],pointVec));  
    
    return DLSUCCESSED;
    
}

int LiteHrnetModel::PostProcess(cv::Mat &out,std::vector<cv::Point> &pointVec)
{    

    int num_joints = out.size[1];
    int height = out.size[2];
    int weight = out.size[3];

    pointVec.resize(num_joints);
    try{
    
        cv::parallel_for_(cv::Range(0, num_joints), [&](const cv::Range& r) {
            for (int i = r.start; i < r.end; i++) {
                uchar* src_data = out.data + i * height * weight* sizeof(float);
                cv::Mat mat = cv::Mat(height, weight, CV_32FC1, src_data);
                double  max_val;
                cv::Point  max_loc;
                cv::minMaxLoc(mat, nullptr, &max_val, nullptr, &max_loc);		
                if (max_val > 0.0) 
                {
                    pointVec[i] = max_loc*stride_; 
                }
                else
                {
                    pointVec[i] = cv::Point(-1,-1);
                }
            }
        });

    }catch (cv::Exception &e)
		{
			// output exception information
			LOG(ERROR) << "message: " << e.what();
			return DLFAILED;
		}


    return DLSUCCESSED;
}

int LiteHrnetModel::DeInitModel()
{
    if(model_ != nullptr)
    {
        model_->DeInitModel();
    }
    
    return DLSUCCESSED;
}


}