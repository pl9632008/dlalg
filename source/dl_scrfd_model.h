#ifndef DL_SCRFD_MODEL_H
#define DL_SCRFD_MODEL_H

#include "dl_model.h"
#include "dl_format.h"
#include "dl_algorithm.h"
#include <opencv2/opencv.hpp>


namespace yjh_deeplearning{




class SCRFDModel{
    public:
        SCRFDModel(dlalg_jsons::ModelInfo &modelInfo);
        ~SCRFDModel()=default;
        int InitModel();
        int ModelPredict(const cv::Mat &img, AIOutputInfo &output_info); 
        void DeInitModel();
    
    private:
        int PreProcess(cv::Mat &img);
        int PostProcess(std::vector<cv::Mat> &outs,AIOutputInfo &output_info);

        std::shared_ptr<BaseModelPredictor> model_;
        std::vector<dlalg_jsons::PreprocessInfo> preprocess_list_;

        float conf_threshold_;// 置信度阈值
        int img_height_,img_width_;
		std::vector<float> stride_;	
		float iou_threshold_;

        std::string infer_engine_;

        std::vector<cv::Mat> out_tensors_;     
		int newh_,neww_, padh_, padw_;
		float ratioh_,ratiow_;
 
 };
     
  
    
}



#endif