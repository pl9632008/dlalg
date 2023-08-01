#ifndef DL_ALG_OCR_H
#define DL_ALG_OCR_H

#include "dl_model.h"
#include "dl_algfactory.h"
#include "dl_crnn_model.h"
#include "dl_dbnet_model.h"

#include <opencv2/opencv.hpp>
#include <memory>

namespace yjh_deeplearning{



class OCRDLAlg :public BaseDLAlg {
	public:      
		int Init(dlalg_jsons::AlgInfo &algInfo);
		int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info);     
		int DeInit();
	private:
   

		std::shared_ptr<DbnetModel> det_model_{nullptr};
		std::shared_ptr<CRNNModel> rec_model_{nullptr};

		std::vector<std::string> class_name_;
		

		// std::shared_ptr<BaseModelPredictor> angle_model_;

		// int model1_height_{0};
        // int model1_width_{0};
		// bool auto_preprocess1_{false};
		// cv::Point classNumber_;
    	// double classProb_;
		// std::vector<cv::Mat> out_tensors_;
       
        std::vector<TextBox> det_boxes_;

    
};


}

#endif
