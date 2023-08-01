#ifndef DL_YOLOV5_MODEL_H
#define DL_YOLOV5_H

#include "dl_model.h"
#include "dl_format.h"
#include "dl_util.h"

#include <opencv2/opencv.hpp>


namespace yjh_deeplearning{




class Yolov5Model{
    public:
        Yolov5Model(dlalg_jsons::ModelInfo &modelInfo);
        ~Yolov5Model()=default;
        int InitModel();
        int ModelPredict(const cv::Mat &img, std::vector<DetectionObj> &detection_result); 
        int ModelPredict(const std::vector<cv::Mat> &imgs, std::vector<std::vector<DetectionObj>> &detection_result); 
        void DeInitModel();
 
    
    private:

        std::shared_ptr<BaseModelPredictor> model_;
        std::vector<dlalg_jsons::PreprocessInfo> preprocess_list_;

        int PreProcess(cv::Mat &img);
        int PostProcess(const cv::Mat &out_tensor,std::vector<std::vector<DetectionObj>> &detection_results);
        void CalcOirDet(const cv::Mat& img, DetectionObj &obj);

        float conf_threshold_;// 置信度阈值
        float iou_threshold_;// NMS阈值
        float obj_threshold_;// 目标置信度值
        bool multi_label_;
        bool auto_preprocess_{false};

        std::string infer_engine_;
      
        std::vector<cv::Mat> out_tensors_;
        std::vector<int> classIds_;
        std::vector<float> confidences_;
        std::vector<cv::Rect2d> boxes_;
 
 };
     
  
    
}



#endif