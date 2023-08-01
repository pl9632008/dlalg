#include "dl_dbnet_model.h"
#include "dl_common.h"
#include "dl_util.h"
#include "dl_opencvdnn.h"
#include "dl_onnxruntime.h"
#include "dl_tensorrt.h"
#include <glog/logging.h>


namespace yjh_deeplearning{

DbnetModel::DbnetModel(dlalg_jsons::ModelInfo &modelInfo)
{
    preprocess_list_ = modelInfo.preprocess_list;  
    infer_engine_ = modelInfo.infer_engine;

    auto conf_iter = modelInfo.other_conf_thresh.find("box_score_thresh");
    if(conf_iter != modelInfo.other_conf_thresh.end())
    {
        box_score_thresh_ = conf_iter->second;
    }

    conf_iter = modelInfo.other_conf_thresh.find("scale_down_ratio");
    if(conf_iter != modelInfo.other_conf_thresh.end())
    {
        scale_down_ratio_ = conf_iter->second;
    }

    conf_iter = modelInfo.other_conf_thresh.find("box_thresh");
    if(conf_iter != modelInfo.other_conf_thresh.end())
    {
        box_thresh_ = conf_iter->second;
    }

    conf_iter = modelInfo.other_conf_thresh.find("un_clip_ratio");
    if(conf_iter != modelInfo.other_conf_thresh.end())
    {
        un_clip_ratio_ = conf_iter->second;
    }

    if(modelInfo.auto_preprocess.size() != 0)
    {
        auto_preprocess_ = true;
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

int DbnetModel::InitModel()
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



int DbnetModel::PreProcess(cv::Mat &img)
{
    try{ 
        
        img_height_ = img.rows;
        img_width_ = img.cols;

        int resize_size = preprocess_list_[0].short_size;
        resize_size = resize_size - resize_size%32;
             
        if(img_height_ < img_width_)
        {
            input_height_ = resize_size;
            input_width_ = img_width_*(1.0)*input_height_/img_height_;
            input_width_ = input_width_ - input_width_%32;
            input_width_ = std::max(32,input_width_);            
        }
        else
        {
            input_width_ = resize_size;
            input_height_ = img_height_*(1.0)*input_width_/img_width_;
            input_height_ = input_height_ - input_height_%32;
            input_height_ = std::max(32,input_height_);
        }

        if(input_width_ > preprocess_list_[0].max_size || input_height_ > preprocess_list_[0].max_size)
        {
            LOG(ERROR)<<"after resize, img size greater max size";
            return DLFAILED;
        }
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::resize(img, img, cv::Size(input_width_,input_height_), 0, 0, cv::INTER_LINEAR);
     
        img.convertTo(img, CV_32F, 1.0 / preprocess_list_[0].img_max_value);
		cv::subtract(img, cv::Scalar(preprocess_list_[0].img_mean[0], preprocess_list_[0].img_mean[1], preprocess_list_[0].img_mean[2]), img);
		cv::divide(img, cv::Scalar(preprocess_list_[0].img_std[0], preprocess_list_[0].img_std[1], preprocess_list_[0].img_std[2]), img);     
        
    }
    catch (cv::Exception& e)
    {
        // output exception information
        LOG(ERROR) << "message: " << e.what() ;
        return YJH_AI_INPUT_IMGPREPROCESS_ERROR;
    }
    return DLSUCCESSED;
}

int DbnetModel::ModelPredict(const cv::Mat &img, std::vector<TextBox> &output_boxes)
{      
    cv::Mat image = img.clone();  
    if(auto_preprocess_ == false)
    {  
        CHECK_SUCCESS(PreProcess(image));
    } 
     
    CHECK_SUCCESS(model_->Predict(image,out_tensors_));
      
    CHECK_SUCCESS(PostProcess(out_tensors_[0],output_boxes));        
    
        
    return DLSUCCESSED;
    
}

int DbnetModel::PostProcess(cv::Mat &out, std::vector<TextBox> &output_boxes)
{    
   
    try{
        std::vector<int> newshape{out.size[2], out.size[3]};      
        cv::Mat out_img = out.reshape(1,newshape);      
     
        ratio_width_ = input_width_*1.0/img_width_;
        ratio_height_ = input_height_*1.0/img_height_;
        output_boxes.clear();
        cv::Mat norfMapMat = out_img > box_thresh_;
        // find rs boxes
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contours.size(); ++i) {
            float minSideLen, perimeter;
            std::vector<cv::Point> minBox = getMinBoxes(contours[i], minSideLen, perimeter);
            if (minSideLen < minArea_)
                continue;
            float score = boxScoreFast(out_img, contours[i]);
            if (score < box_score_thresh_)
                continue;

            std::vector<cv::Point> clipBox = unClip(minBox, perimeter, un_clip_ratio_);
            std::vector<cv::Point> clipMinBox = getMinBoxes(clipBox, minSideLen, perimeter);

            if (minSideLen < minArea_ + 2)
                continue;

            for (int j = 0; j < clipMinBox.size(); ++j) {
                clipMinBox[j].x = (clipMinBox[j].x / ratio_width_);
                clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), img_width_);

                clipMinBox[j].y = (clipMinBox[j].y / ratio_height_);
                clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), img_height_);
            }
           
            output_boxes.emplace_back(TextBox{clipMinBox,  score});
        }
        std::reverse(output_boxes.begin(), output_boxes.end());     
    

    }catch (cv::Exception &e)
		{
			// output exception information
			LOG(ERROR) << "message: " << e.what();
			return DLFAILED;
		}


    return DLSUCCESSED;
}

void DbnetModel::DeInitModel()
{
    if(model_ != nullptr)
    {
        model_->DeInitModel();
    }
}



}