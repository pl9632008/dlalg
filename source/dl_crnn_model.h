#ifndef DL_CRNN_MODEL_H
#define DL_CRNN_MODEL_H

#include "dl_model.h"
#include "dl_format.h"
#include "dl_algorithm.h"
#include <opencv2/opencv.hpp>


namespace yjh_deeplearning{




class CRNNModel{
    public:
        CRNNModel(dlalg_jsons::ModelInfo &modelInfo);
        ~CRNNModel()=default;
        int InitModel();
        int ModelPredict(const cv::Mat &img, std::string &res_str); 
        void DeInitModel();
    
    private:
        int PreProcess(cv::Mat &img);
        int PostProcess(cv::Mat &out,std::string &res_str);

        std::shared_ptr<BaseModelPredictor> model_;
        std::vector<dlalg_jsons::PreprocessInfo> preprocess_list_;

        int img_height_,img_width_;
        bool auto_preprocess_{false};
        std::string infer_engine_;
        std::string keys_path_;

        std::vector<std::string> labelMapping_{};
        std::vector<cv::Mat> out_tensors_;     
        
	
 
 };
     
  
    
}



#endif