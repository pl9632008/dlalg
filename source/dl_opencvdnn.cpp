#include "dl_opencvdnn.h"



// #include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>
#include <algorithm>

namespace yjh_deeplearning
{

	OpencvDNNPredictor::OpencvDNNPredictor(dlalg_jsons::ModelInfo &modelInfo):BaseModelPredictor(modelInfo)
	{
		
	}


	int OpencvDNNPredictor::InitModel()
	{
		try
		{
			if (model_type_ == "ONNX")
			{
				net_ = cv::dnn::readNet(weight_path_);
			}
			else if (model_type_ == "Caffe")
			{
				net_ = cv::dnn::readNetFromCaffe(cfg_path_, weight_path_);
			}
			else if (model_type_ == "Darknet")
			{
				net_ = cv::dnn::readNetFromDarknet(cfg_path_, weight_path_);
			}
			else if (model_type_ == "Torch")
			{
				net_ = cv::dnn::readNetFromTorch(weight_path_);
			}
			else
			{
				LOG(ERROR) << "opencv model type error";
				return YJH_AI_OPENCV_TYPE_ERROR;
			}
		}
		catch (cv::Exception &e)
		{
			// output exception information
			LOG(ERROR) << "message: " << e.what();
			return YJH_AI_OPENCV_INIT_ERROR;
		}

		try
		{
			if (gpu_index_ >= 0)
			{
				cv::cuda::setDevice(gpu_index_);
				net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
				net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
			}
			else
			{
				net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
				net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
			}
			if (name_output_.empty())
			{
				const std::vector<std::string> &out_names = net_.getUnconnectedOutLayersNames();
				for (unsigned i = 0; i < out_names.size(); i++)
				{
					name_output_.emplace_back(out_names[i]);
				}
			}
		}
		catch (cv::Exception &e)
		{
			// output exception information
			LOG(ERROR) << "message: " << e.what();
			return YJH_AI_OPENCV_INIT_ERROR;
		}

		return DLSUCCESSED;
	}

	int OpencvDNNPredictor::PredictImp(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs)
	{
		try
		{
			net_.setInput(cv::dnn::blobFromImage(img));
			net_.forward(outputBlobs, name_output_);
		}
		catch (cv::Exception &e)
		{
			// output exception information
			LOG(ERROR) << "message: " << e.what();
			return YJH_AI_OPENCV_INFERENCE_ERROR;
		}
		return DLSUCCESSED;
	}

	int OpencvDNNPredictor::PredictImp(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs)
	{	
		try
		{	
			if(imgs.size() != name_input_.size())
			{
				LOG(ERROR) << "input mat num error!";
				return YJH_AI_OPENCV_INFERENCE_ERROR;
			}
			for (unsigned int i = 0; i < imgs.size(); i++)
			{
				net_.setInput(cv::dnn::blobFromImage(imgs[i]),name_input_[i]);
			}
			net_.forward(outputBlobs, name_output_);
		}
		catch (cv::Exception &e)
		{
			// output exception information
			LOG(ERROR) << "message: " << e.what();
			return YJH_AI_OPENCV_INFERENCE_ERROR;
		}
		return DLSUCCESSED;
	}


}