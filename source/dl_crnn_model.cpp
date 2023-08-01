#include "dl_crnn_model.h"
#include "dl_common.h"
#include "dl_opencvdnn.h"
#include "dl_onnxruntime.h"
#include "dl_tensorrt.h"
#include <glog/logging.h>


namespace yjh_deeplearning{

CRNNModel::CRNNModel(dlalg_jsons::ModelInfo &modelInfo)
{
    preprocess_list_ = modelInfo.preprocess_list;  
    infer_engine_ = modelInfo.infer_engine;
    if(modelInfo.auto_preprocess.size() != 0)
    {
        auto_preprocess_ = true;
    }
    auto file_iter = modelInfo.other_conf.find("keys_path");
    if(file_iter != modelInfo.other_conf.end())
    {
        keys_path_ = file_iter->second;
    }
    if(infer_engine_ == INFER_ENGINE_ORT)
    {
        model_ = std::make_shared<ONNXRuntimePredictor>(modelInfo); 
    }    
    else
    {
        model_ = nullptr;
    }
        
      
 }

int CRNNModel::InitModel()
{

    if(preprocess_list_.size() == 0 || model_ == nullptr)
    {
        LOG(ERROR) << "preprocess config error or model is nullptr";
        return YJH_AI_PARAMETER_ERROR;
    }

    std::ifstream ifs(keys_path_,std::ios::in);
    if(!ifs.is_open())
    {   
        LOG(ERROR) << "can not open crnn key file "<<keys_path_;
        return YJH_AI_CFGFILE_NOFIND_ERROR;        
    }
    labelMapping_.emplace_back("");
    std::string str;
    while(!ifs.eof())//eof() 检查是否到达文件末尾
    {        
        std::getline(ifs,str);       
        labelMapping_.emplace_back(str);
    }
    

    img_height_ = preprocess_list_[0].img_height;
    img_width_ = preprocess_list_[0].img_width;
    return model_->InitModel();
}


int CRNNModel::ModelPredict(const cv::Mat &img, std::string &res_str)
{      
    cv::Mat image = img.clone();  
    if(auto_preprocess_ == false)
    {        
        CHECK_SUCCESS(PreProcess(image));
    } 
    
    CHECK_SUCCESS(model_->Predict(image,out_tensors_));
      
    CHECK_SUCCESS(PostProcess(out_tensors_[0],res_str));        
    
    
    return DLSUCCESSED;
    
}

int CRNNModel::PreProcess(cv::Mat &img)
{
    try{      
        cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
        float cur_ratio = (img.cols*1.0) / (img.rows*1.0);
        int w = preprocess_list_[0].img_width;
        int h = preprocess_list_[0].img_height;
        if(cur_ratio <  w*1.0/h)
        {
            w = h*(1.0)*cur_ratio;
        }
        
        cv::Mat re(h, w, CV_8UC1);
        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

        cv::Mat out(preprocess_list_[0].img_height, preprocess_list_[0].img_width, CV_8UC1, cv::Scalar(0));
        re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));        
        img = out;       
        img.convertTo(img, CV_32F,1.0/preprocess_list_[0].img_max_value);        
    }
    catch (cv::Exception& e)
    {
        // output exception information
        LOG(ERROR) << "message: " << e.what() ;
        return YJH_AI_INPUT_IMGPREPROCESS_ERROR;
    }
    return DLSUCCESSED;
}


// str_pred = []
//             for p in pred_idx:
//                 if p != last_p and p != 0:
//                     str_pred.append(self.labelMapping[p])
//                 last_p = p
//             final_str = ''.join(str_pred)
//             final_str_list.append(final_str)

int CRNNModel::PostProcess(cv::Mat &out,std::string &res_str)
{    
    int p;
    int last_p=0;
    res_str = "";
    try{
        for(int i=0;i<out.size[1];i++)
        {
            p = out.at<int>(0,i);
            if(p != last_p && p != 0)
            {
                res_str = res_str+labelMapping_[p];
            }
            last_p = p;            
        }

    }catch (cv::Exception &e)
		{
			// output exception information
			LOG(ERROR) << "message: " << e.what();
			return DLFAILED;
		}


    return DLSUCCESSED;
}


void CRNNModel::DeInitModel()
{
    if(model_ != nullptr)
    {
        model_->DeInitModel();
    }
}

}