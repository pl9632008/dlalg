#ifndef DL_ALG_KEYPOINT_H
#define DL_ALG_KEYPOINT_H

#include "dl_model.h"
#include "dl_algfactory.h"
#include "dl_yolov5_model.h"
#include "dl_litehrnet_model.h"

#include <opencv2/opencv.hpp>
#include <memory>

namespace yjh_deeplearning{



class KeypointDLAlg :public BaseDLAlg {
	public:      
		int Init(dlalg_jsons::AlgInfo &algInfo);
		int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info); 
        int DeInit(); 
		
	private:

     	std::shared_ptr<Yolov5Model> detection_model_;
		std::shared_ptr<LiteHrnetModel> keypoint_model_;
        std::vector<std::string> class_name_;

        int keypoint_height_{0};
        int keypoint_width_{0};
               
        float expanding_{1.25};
        std::vector<DetectionObj> detection_results_;

        cv::Mat inverse_warp_mat_;

};


}

#endif
