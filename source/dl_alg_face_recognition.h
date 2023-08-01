#ifndef DL_ALG_FACE_RECOGNITION_H
#define DL_ALG_FACE_RECOGNITION_H

#include "dl_model.h"
#include "dl_scrfd_model.h"
#include "dl_algfactory.h"

#include <opencv2/opencv.hpp>
#include <memory>

namespace yjh_deeplearning{

//标准的关键点。
float src_align[5][2] = {
    {30.2946f, 51.6963f},
    {65.5318f, 51.5014f},
    {48.0252f, 71.7366f},
    {33.5493f, 92.3655f},
    {62.7299f, 92.2041f}};

#define STANDARDWIDTH 112

class FaceRecognitionDLAlg :public BaseDLAlg {
	public:      
		int Init(dlalg_jsons::AlgInfo &algInfo);
		int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info);
		int DeInit();   
		
	private:

		std::shared_ptr<SCRFDModel> scrfd_model_;
		std::shared_ptr<BaseModelPredictor> mobileface_model_;	
        std::vector<cv::Mat> out_tensors_;
		
		cv::Mat src_mat_align_;
		cv::Mat dst_mat_align_;
		float dst_point_[10];
		cv::Mat o_img_;
		cv::Mat face_roi_;
		

};


}

#endif
