#include "dl_orientedrcnn_model.h"
#include "dl_common.h"
#include "dl_tensorrt.h"

#include <glog/logging.h>

namespace yjh_deeplearning{



OrientedRcnnModel::OrientedRcnnModel(dlalg_jsons::ModelInfo &modelInfo)
{   
    preprocess_list_ = modelInfo.preprocess_list;
    model_ = std::make_shared<TenorrtPredictor>(modelInfo);    
    conf_threshold_ = modelInfo.conf_threshold;  
 }

int OrientedRcnnModel::InitModel()
{
    if(preprocess_list_.size() == 0 || model_ == nullptr)
    {
        LOG(ERROR) << "preprocess config error or model is nullptr";
        return YJH_AI_PARAMETER_ERROR;
    }
    return model_->InitModel();
}

int OrientedRcnnModel::ModelPredict(const cv::Mat &src_img, std::vector<DetectionObj> &detection_results)
{
    cv::Mat image;   
    try{
        if(src_img.cols == preprocess_list_[0].img_width && src_img.rows == preprocess_list_[0].img_height)
        {
            image = src_img.clone();
        }
        else
        {
            image = src_img;
            int w, h, x, y;
            float r_w = preprocess_list_[0].img_width / (image.cols*1.0);
            float r_h = preprocess_list_[0].img_height / (image.rows*1.0);
            if (r_h > r_w) {
                w = preprocess_list_[0].img_width;
                h = r_w * image.rows;
                x = 0;
                y = (preprocess_list_[0].img_height - h) / 2;
            } else {
                w = r_h * image.cols;
                h = preprocess_list_[0].img_height;
                x = (preprocess_list_[0].img_width - w) / 2;
                y = 0;
            }
            cv::Mat re(h, w, CV_8UC3);
            cv::resize(image, re, re.size(), 0, 0, cv::INTER_LINEAR);
            cv::Mat out(preprocess_list_[0].img_height, preprocess_list_[0].img_width, CV_8UC3, cv::Scalar(128, 128, 128));
            re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));        
            image = out;           
        }                
		    
		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        image.convertTo(image, CV_32F);   
        cv::subtract(image, cv::Scalar(preprocess_list_[0].img_mean[0], preprocess_list_[0].img_mean[1], preprocess_list_[0].img_mean[2]), image);      
		cv::divide(image, cv::Scalar(preprocess_list_[0].img_std[0], preprocess_list_[0].img_std[1], preprocess_list_[0].img_std[2]), image);    
	}
	catch (cv::Exception& e)
    {
        // output exception information
        LOG(ERROR) << "message: " << e.what() ;
        return YJH_AI_INPUT_IMGPREPROCESS_ERROR;
    }  
    CHECK_SUCCESS(model_->BatchPredict({{image}},out_tensors_));  
    float ratioh = float(src_img.rows) / preprocess_list_[0].img_height;
    float ratiow = float(src_img.cols) / preprocess_list_[0].img_width;
    
    detection_results.clear(); 
    DetectionObj result;
   
   
    for(int j=0;j<out_tensors_[0].size[1];j++) 
    {    
        if(out_tensors_[0].at<float>(0,j,5) < conf_threshold_)
        {
            continue;
        }
        // LOG(ERROR)<< out_tensors_[1].at<int>(0,j,0);
        // LOG(ERROR)<< out_tensors_[0].at<float>(0,j,0) <<" "<<out_tensors_[0].at<float>(0,j,1)<<" "<<out_tensors_[0].at<float>(0,j,2)<<" "<<out_tensors_[0].at<float>(0,j,3)<<" "<<out_tensors_[0].at<float>(0,j,4)<<" "<<out_tensors_[0].at<float>(0,j,5);
        result.class_idx=int(out_tensors_[1].at<int>(0,j,0));  
        result.score =  out_tensors_[0].at<float>(0,j,5);  
        result.center_x = out_tensors_[0].at<float>(0,j,0)*ratiow;
        result.center_y = out_tensors_[0].at<float>(0,j,1)*ratioh;
        result.width = out_tensors_[0].at<float>(0,j,2)*ratiow;
        result.height = out_tensors_[0].at<float>(0,j,3)*ratioh;
        result.angle = out_tensors_[0].at<float>(0,j,4);  
        detection_results.emplace_back(result);
    }
    
	return DLSUCCESSED;

}

void OrientedRcnnModel::DeInitModel()
{       
    if(model_ != nullptr)
    {
        model_->DeInitModel();
    }
       
}


}