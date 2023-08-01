#ifndef DL_ORIENTEDRCNN_MODEL_H
#define DL_ORIENTEDRCNN_MODEL_H

#include "dl_model.h"
#include "dl_format.h"
#include "dl_util.h"

#include <opencv2/opencv.hpp>


namespace yjh_deeplearning{




class OrientedRcnnModel{
    public:
        OrientedRcnnModel(dlalg_jsons::ModelInfo &modelInfo);
        ~OrientedRcnnModel()=default;
        int InitModel();
        int ModelPredict(const cv::Mat &img, std::vector<DetectionObj> &detection_result); 
        void DeInitModel();
    
    private:

        float conf_threshold_;// 置信度阈值
          
        std::shared_ptr<BaseModelPredictor> model_;
        std::vector<cv::Mat> out_tensors_;

        std::vector<dlalg_jsons::PreprocessInfo> preprocess_list_;
                
             


 
 };
     
  
    
}



#endif