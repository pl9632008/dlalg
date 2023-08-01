#ifndef DL_ALG_SCRFD_H
#define DL_ALG_SCRFD_H

#include "dl_scrfd_model.h"
#include "dl_algfactory.h"


#include <opencv2/opencv.hpp>
#include <memory>

namespace yjh_deeplearning{



class SCRFDDLAlg :public BaseDLAlg {
	public:      
		int Init(dlalg_jsons::AlgInfo &algInfo);
		int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info);
		int DeInit();   

	private:      

		std::shared_ptr<SCRFDModel> scrfd_model_;
		
		
};


}

#endif
