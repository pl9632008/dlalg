#ifndef DL_ALG_WENT_H
#define DL_ALG_WENT_H

#include "dl_model.h"
#include "dl_algfactory.h"


#include <opencv2/opencv.hpp>
#include <memory>

namespace yjh_deeplearning{



class U2NetDLAlg :public BaseDLAlg {
	public:      
		int Init(dlalg_jsons::AlgInfo &algInfo);
		int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info);
		int DeInit();        

	private:
		std::shared_ptr<BaseModelPredictor> wentie_model_;
        std::vector<cv::Mat> out_tensors_;
        cv::Mat o_img_;

};


}

#endif
