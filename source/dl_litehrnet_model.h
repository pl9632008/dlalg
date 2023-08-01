#ifndef DL_LITEHRNET_MODEL_H
#define DL_LITEHRNET_MODEL_H

#include "dl_model.h"
#include "dl_format.h"
#include "dl_algorithm.h"
#include <opencv2/opencv.hpp>


namespace yjh_deeplearning{




class LiteHrnetModel{
    public:
        LiteHrnetModel(dlalg_jsons::ModelInfo &modelInfo);
        ~LiteHrnetModel()=default;
        int InitModel();
        int ModelPredict(const cv::Mat &img, std::vector<cv::Point> &pointVec); 
        int DeInitModel();
    
    private:
      
        int PostProcess(cv::Mat &out,std::vector<cv::Point> &pointVec);

        std::shared_ptr<BaseModelPredictor> model_;
        std::vector<dlalg_jsons::PreprocessInfo> preprocess_list_;

        int img_height_,img_width_;
        bool auto_preprocess_{false};

        std::string infer_engine_;
        int stride_{4};

        std::vector<cv::Mat> out_tensors_;     
	
 
 };
     
  
    
}



#endif