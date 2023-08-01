#ifndef DL_ALG_CLS_H
#define DL_ALG_CLS_H

#include "dl_model.h"
#include "dl_algfactory.h"

#include <opencv2/opencv.hpp>
#include <memory>

namespace yjh_deeplearning{



class ClsDLAlg :public BaseDLAlg {
	public:      
		int Init(dlalg_jsons::AlgInfo &algInfo);
		int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info);     
		
	private:
		std::shared_ptr<BaseModelPredictor> cls_model_;	
        std::vector<cv::Mat> out_tensors_;
		std::vector<std::string> class_name_;
		bool auto_preprocess_{false};
	
		cv::Point classNumber_;
    	double classProb_;
		

};


}

#endif
