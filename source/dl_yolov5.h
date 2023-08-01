#ifndef DL_YOLOV5_H
#define DL_YOLOV5_H

#include "dl_yolov5_model.h"
#include "dl_format.h"
#include "dl_algfactory.h"

#include <opencv2/opencv.hpp>


namespace yjh_deeplearning{





class Yolov5Alg:public BaseDLAlg{
    public:
        
        int Init(dlalg_jsons::AlgInfo &algInfo);
		int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info); 
        int ProcessPic(const std::vector<AIInputInfo> &input_list,std::vector<AIOutputInfo> &output_list);
        int DeInit();
        // int LibTorchInit();
        // int LibTorchDetect(); 
    
    private:

        std::shared_ptr<Yolov5Model> yolov5_model_;
        std::vector<std::string> class_name_;
        std::map<std::string,float> class_thresh_;
        std::vector<DetectionObj> detection_results_;
        std::vector<std::vector<DetectionObj>> total_detection_results_;
 
 };
     
  
    
}



#endif