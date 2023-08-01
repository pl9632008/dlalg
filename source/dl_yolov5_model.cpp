#include "dl_yolov5_model.h"
#include "dl_common.h"
#include "dl_opencvdnn.h"
#include "dl_tensorrt.h"
#include "dl_ascendcl.h"
#include <glog/logging.h>

namespace yjh_deeplearning{



Yolov5Model::Yolov5Model(dlalg_jsons::ModelInfo &modelInfo)
{       
    preprocess_list_ = modelInfo.preprocess_list;
    conf_threshold_ = modelInfo.conf_threshold;
    iou_threshold_ = modelInfo.iou_threshold;
    obj_threshold_ = modelInfo.obj_threshold;
    multi_label_ = modelInfo.multi_label;
    infer_engine_ = modelInfo.infer_engine;
    if(modelInfo.auto_preprocess.size() != 0)
    {
        auto_preprocess_ = true;
    }
    if(infer_engine_ == INFER_ENGINE_OPENCV)
    {
         model_ = std::make_shared<OpencvDNNPredictor>(modelInfo);  
    }
    else if(infer_engine_ == INFER_ENGINE_TRT)
    {
         model_ = std::make_shared<TenorrtPredictor>(modelInfo);  
    }
    else if(infer_engine_ == INFER_ENGINE_ASCENDCL)
    {
         model_ = std::make_shared<AscendCLPredictor>(modelInfo);  
    }
    else
    {
        model_ = nullptr;
    }
        
   
 }

int Yolov5Model::InitModel()
{
    if(preprocess_list_.size() == 0 || model_ == nullptr)
    {
        LOG(ERROR) << "preprocess config error or model is nullptr";
        return YJH_AI_PARAMETER_ERROR;
    }
    
    return model_->InitModel();
  
}

int Yolov5Model::ModelPredict(const cv::Mat &src_img, std::vector<DetectionObj> &detection_results)
{
    cv::Mat image; 
    // LOG(ERROR)<<"autol_preprocess_: "<<auto_preprocess_;
 
    try{ 		
        image = src_img;
        if(auto_preprocess_ == false)
        {          
            CHECK_SUCCESS(PreProcess(image));
        }
          
	}
	catch (cv::Exception& e)
    {
        // output exception information
        LOG(ERROR) << "message: " << e.what() ;
        return YJH_AI_INPUT_IMGPREPROCESS_ERROR;
    }
    
    std::vector<std::vector<DetectionObj>> total_detection_results;
    
    CHECK_SUCCESS(model_->Predict(image,out_tensors_));       
    PostProcess(out_tensors_[out_tensors_.size()-1],total_detection_results);    

    detection_results.swap(total_detection_results[0]);

   
    for(unsigned int i=0;i<detection_results.size();i++)
    {
        CalcOirDet(src_img,detection_results[i]);      
    }
   
	return DLSUCCESSED;
}

void Yolov5Model::CalcOirDet(const cv::Mat& img, DetectionObj &obj)
{
    float r_w =  (img.cols * 1.0)/preprocess_list_[0].img_width ;
    float r_h =  (img.rows * 1.0)/preprocess_list_[0].img_height ;
    float ratio;
    if(r_w > r_h)
    {        
        obj.center_y = obj.center_y - (preprocess_list_[0].img_height-img.rows/r_w)/2.0;
        ratio  = r_w;
    }
    else
    {
        obj.center_x = obj.center_x - (preprocess_list_[0].img_width-img.cols/r_h)/2.0;
        ratio  = r_h;
    }
    
    obj.center_x *= ratio;   
    obj.center_y *= ratio;
    obj.width *= ratio;
    obj.height *= ratio;
}


int Yolov5Model::PreProcess(cv::Mat &img)
{
    try{ 
        int w, h, x, y;
        float r_w = preprocess_list_[0].img_width / (img.cols*1.0);
        float r_h = preprocess_list_[0].img_height / (img.rows*1.0);
        if (r_h > r_w) {
            w = preprocess_list_[0].img_width;
            h = r_w * img.rows;
            x = 0;
            y = (preprocess_list_[0].img_height - h) / 2;
        } else {
            w = r_h * img.cols;
            h = preprocess_list_[0].img_height;
            x = (preprocess_list_[0].img_width - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat out(preprocess_list_[0].img_height, preprocess_list_[0].img_width, CV_8UC3, cv::Scalar(128, 128, 128));
        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));        
        img = out;
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
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

int Yolov5Model::PostProcess(const cv::Mat &out_tensor,std::vector<std::vector<DetectionObj>> &detection_results)
{
       
  
    float max_class_socre, class_socre;
    int max_class_id;
    detection_results.clear();   
    for(int i=0;i<out_tensor.size[0];i++)
    {
        classIds_.clear();
        confidences_.clear();
        boxes_.clear();
        std::vector<DetectionObj> si_detection{};
        for(int j=0;j<out_tensor.size[1];j++) 
        {
            if(out_tensor.at<float>(i,j,4) > obj_threshold_)
            {                 
                if(multi_label_ == true)
                {
                    for (int c = 0; c < out_tensor.size[2]-5; ++c) //// get max socre
                    {     
                        class_socre = out_tensor.at<float>(i,j,c+5)*out_tensor.at<float>(i,j,4);                
                        if (class_socre > conf_threshold_)
                        {
                            classIds_.push_back(c);           
                            confidences_.push_back(class_socre);
                            // LOG(ERROR)<<out_tensor.at<float>(i,j,0)<<" "<<out_tensor.at<float>(i,j,1)<<" "<<out_tensor.at<float>(i,j,2)<<" "<<out_tensor.at<float>(i,j,3);
                            boxes_.push_back( cv::Rect2d((out_tensor.at<float>(i,j,0)-out_tensor.at<float>(i,j,2)/2)+c*preprocess_list_[0].img_width,(out_tensor.at<float>(i,j,1)-out_tensor.at<float>(i,j,3)/2)+c*preprocess_list_[0].img_height,out_tensor.at<float>(i,j,2),out_tensor.at<float>(i,j,3)));
                        }
                    } 
                }
                else
                {
                    max_class_socre = 0;                   
                    max_class_id = 0;                 
                    for (int c = 0; c < out_tensor.size[2]-5; ++c) //// get max socre
                    {   
                        class_socre = out_tensor.at<float>(i,j,c+5)*out_tensor.at<float>(i,j,4);                     
                        if (class_socre > max_class_socre)
                        {
                            max_class_socre = class_socre;
                            max_class_id = c;
                        }
                    }
                    // LOG(ERROR)<<out_tensor.at<float>(i,j,0)<<" "<<out_tensor.at<float>(i,j,1)<<" "<<out_tensor.at<float>(i,j,2)<<" "<<out_tensor.at<float>(i,j,3);     
                    classIds_.push_back(max_class_id);           
                    confidences_.push_back(max_class_socre);
                    boxes_.push_back( cv::Rect2d((out_tensor.at<float>(i,j,0)-out_tensor.at<float>(i,j,2)/2)+max_class_id*preprocess_list_[0].img_width,(out_tensor.at<float>(i,j,1)-out_tensor.at<float>(i,j,3)/2)+max_class_id*preprocess_list_[0].img_height,out_tensor.at<float>(i,j,2),out_tensor.at<float>(i,j,3)));
               }               
            }            
        }

        std::vector<int> indices;      
        cv::dnn::NMSBoxes(boxes_, confidences_, conf_threshold_, iou_threshold_, indices); 
         
        DetectionObj result;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            int idx = indices[i];
            result.class_idx=classIds_[idx];  
            result.score =  confidences_[idx];  
            result.center_x =  boxes_[idx].x -  classIds_[idx]*preprocess_list_[0].img_width+boxes_[idx].width/2;
            result.center_y = boxes_[idx].y -  classIds_[idx]*preprocess_list_[0].img_height+boxes_[idx].height/2;
            result.width = boxes_[idx].width;
            result.height = boxes_[idx].height;           
            si_detection.push_back(result);
        }       
        detection_results.emplace_back(si_detection);        
    }
      
    return DLSUCCESSED;
}

int Yolov5Model::ModelPredict(const std::vector<cv::Mat> &imgs, std::vector<std::vector<DetectionObj>> &total_detection_results)
{
    if(infer_engine_ != INFER_ENGINE_TRT)
    {
        LOG(ERROR) << " infer engin error,only support tensorrt " ;
        return YJH_AI_INPUT_IMGPREPROCESS_ERROR;
    }
    
    cv::Mat image; 
    std::vector<std::vector<cv::Mat>> input_imgs; 
    try{ 
        for(unsigned int i=0;i<imgs.size();i++)
        {
            image = imgs[i];
            if(auto_preprocess_ == false)
            {
                CHECK_SUCCESS(PreProcess(image));
            }                         
            input_imgs.emplace_back(std::vector<cv::Mat>({image.clone()}));
        }
        
	}
	catch (cv::Exception& e)
    {
        // output exception information
        LOG(ERROR) << "message: " << e.what() ;
        return YJH_AI_INPUT_IMGPREPROCESS_ERROR;
    }
    
    CHECK_SUCCESS(model_->BatchPredict(input_imgs,out_tensors_));
    PostProcess(out_tensors_[out_tensors_.size()-1],total_detection_results);    

    for(unsigned int i=0;i<total_detection_results.size();i++)
    {       
        for(unsigned int j=0;j<total_detection_results[i].size();j++)
        {
            CalcOirDet(imgs[i],total_detection_results[i][j]);         
        }

    }
    

	return DLSUCCESSED;

}

void Yolov5Model::DeInitModel()
{
    if(model_ != nullptr)
    {
        model_->DeInitModel();
    }
}




}