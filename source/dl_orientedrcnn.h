#ifndef DL_ORIENTEDRCNN_H
#define DL_ORIENTEDRCNN_H

#include "dl_orientedrcnn_model.h"
#include "dl_format.h"
#include "dl_algfactory.h"

#include <opencv2/opencv.hpp>


namespace yjh_deeplearning{





class OrientedRcnnAlg:public BaseDLAlg{
    public:
        
        int Init(dlalg_jsons::AlgInfo &algInfo);
		int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info); 
        int DeInit();
        // int LibTorchInit();
        // int LibTorchDetect(); 
    
    private:

        
        // void Sigmoid(cv::Mat* out, int length);
        
        std::shared_ptr<OrientedRcnnModel> orientedrcnn_model_;
        std::vector<std::string> class_name_;
        std::vector<DetectionObj> detection_results;
 
 };
     
  
    
}



#endif