#ifndef DL_ALG_JUFANG_H
#define DL_ALG_JUFANG_H

#include "dl_model.h"
#include "dl_algfactory.h"

#include <opencv2/opencv.hpp>
#include <memory>

namespace yjh_deeplearning{



class JufangDLAlg :public BaseDLAlg {
	public:      
		int Init(dlalg_jsons::AlgInfo &algInfo);
		int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info);     
		
	private:
		std::shared_ptr<BaseModelPredictor> jufang_model_detect_normal_;
		std::shared_ptr<BaseModelPredictor> jufang_model_detect_fangdian_;
        std::vector<cv::Mat> out_tensors_;
		std::vector<std::string> class_name_normal_;
		std::vector<std::string> class_name_fangdian_;
		cv::Point classNumber_;
    	double classProb_;
		

};


}

#endif
