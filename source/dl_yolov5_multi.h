#ifndef DL_YOLOV5_MULTI_H
#define DL_YOLOV5_MULTI_H

#include "dl_yolov5_model.h"
#include "dl_format.h"
#include "dl_algfactory.h"

#include <opencv2/opencv.hpp>


namespace yjh_deeplearning{




class Yolov5AlgMulti:public BaseDLAlg{
    public:
    
        int Init(dlalg_jsons::AlgInfo &algInfo);
		int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info); 
        int ProcessPic(const std::vector<AIInputInfo> &input_list,std::vector<AIOutputInfo> &output_list);
        int DeInit();
        // int LibTorchInit();
        // int LibTorchDetect(); 
    
    private:

        std::vector<std::shared_ptr<Yolov5Model>> multi_yolov5_model_{};
        std::vector<std::vector<std::string>> multi_class_name_{};
        std::vector<std::map<std::string,float>> multi_class_thresh_{};
        std::vector<std::vector<DetectionObj>> multi_detection_results_{};
        std::vector<std::vector<std::vector<DetectionObj>>> total_detection_results_{};
 
};
     
  
    
}



#endif