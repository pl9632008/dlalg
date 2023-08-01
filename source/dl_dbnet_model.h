#ifndef DL_DBNET_MODEL_H
#define DL_DBNET_MODEL_H

#include "dl_model.h"
#include "dl_format.h"
#include "dl_util.h"
#include "dl_algorithm.h"
#include <opencv2/opencv.hpp>


namespace yjh_deeplearning{




class DbnetModel{
    public:
        DbnetModel(dlalg_jsons::ModelInfo &modelInfo);
        ~DbnetModel()=default;
        int InitModel();
        int ModelPredict(const cv::Mat &img, std::vector<TextBox> &output_boxes); 
        void DeInitModel();

    
    private:
        int PreProcess(cv::Mat &img);
        int PostProcess(cv::Mat &out,std::vector<TextBox> &output_boxes);

        std::shared_ptr<BaseModelPredictor> model_;
        std::vector<dlalg_jsons::PreprocessInfo> preprocess_list_;

        int img_height_,img_width_;
        int input_height_,input_width_;
        float ratio_height_,ratio_width_;		

        bool auto_preprocess_{false};

        std::string infer_engine_;
        std::vector<cv::Mat> out_tensors_;    


        float minArea_{3};
       
        float box_score_thresh_ {0.3f};
        float scale_down_ratio_ {0.75f};

        float box_thresh_ {0.3f};
    
        float un_clip_ratio_{2.0f};
        
 
 };
     
  
    
}



#endif