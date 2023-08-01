#ifndef DL_ALG_TENSORR_H
#define DL_ALG_TENSORR_H

#include "dl_model.h"
#include "dl_algfactory.h"


#include <opencv2/opencv.hpp>
#include <memory>

namespace yjh_deeplearning{



class TRTDLAlg :public BaseDLAlg {
	public:      
		int Init(dlalg_jsons::AlgInfo &algInfo);
		int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info);    
		int DeInit(); 
		
	private:
		std::shared_ptr<BaseModelPredictor> trt_model_detect_normal_;	
        std::vector<cv::Mat> out_tensors_;
		std::vector<std::string> class_name_;		
		cv::Point classNumber_;
    	double classProb_;
		

};


}

#endif
