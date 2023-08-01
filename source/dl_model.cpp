#include "dl_model.h"
#include "dl_common.h"
#include "dl_transform.h"

#include <glog/logging.h>
#include <algorithm>

namespace yjh_deeplearning
{

	BaseModelPredictor::BaseModelPredictor(dlalg_jsons::ModelInfo &modelInfo)
	{
		preprocess_list_.swap(modelInfo.preprocess_list);		

		weight_path_ = modelInfo.weight_path;
		model_type_ = modelInfo.model_type;
		gpu_index_ = modelInfo.gpu_index;

		is_dynamic_infer = modelInfo.is_dynamic_infer;

		name_input_.swap(modelInfo.name_input);
		name_output_.swap(modelInfo.name_output);
		auto_preprocess_.swap(modelInfo.auto_preprocess);
		batch_size_ = modelInfo.batch_size;

		is_half_ = modelInfo.is_half;
	}

	int BaseModelPredictor::GeneralPreProcess(cv::Mat &image)
	{
		if(preprocess_list_.size() == 0)
		{
			LOG(ERROR) << "preprocess config error";
			return YJH_AI_INPUT_STDORMEAN_ERROR;
		}
		if (preprocess_list_[0].img_mean.size() != GENERAL_STDORMEAN_SIZE && preprocess_list_[0].img_std.size() != GENERAL_STDORMEAN_SIZE)
		{
			LOG(ERROR) << "img_mean or img_std size error";
			return YJH_AI_INPUT_STDORMEAN_ERROR;
		}
		try
		{			
			cv::resize(image, image, cv::Size(preprocess_list_[0].img_width, preprocess_list_[0].img_height), 0, 0, cv::INTER_LINEAR);
			image.convertTo(image, CV_32F, 1.0 / preprocess_list_[0].img_max_value);
			cv::subtract(image, cv::Scalar(preprocess_list_[0].img_mean[0], preprocess_list_[0].img_mean[1], preprocess_list_[0].img_mean[2]), image);
			cv::divide(image, cv::Scalar(preprocess_list_[0].img_std[0], preprocess_list_[0].img_std[1], preprocess_list_[0].img_std[2]), image);
		}
		catch (cv::Exception &e)
		{
			// output exception information
			LOG(ERROR) << "message: " << e.what();
			return YJH_AI_INPUT_IMGPREPROCESS_ERROR;
		}
		return DLSUCCESSED;
	}

	int BaseModelPredictor::PreProcessByMulMaxValue(cv::Mat &image)
	{
		if(preprocess_list_.size() == 0)
		{
			LOG(ERROR) << "preprocess config error";
			return YJH_AI_INPUT_STDORMEAN_ERROR;
		}
		if (preprocess_list_[0].img_mean.size() != GENERAL_STDORMEAN_SIZE && preprocess_list_[0].img_std.size() != GENERAL_STDORMEAN_SIZE)
		{
			LOG(ERROR) << "img_mean or img_std size error";
			return YJH_AI_INPUT_STDORMEAN_ERROR;
		}
		try
		{
			image.convertTo(image, CV_32F, 1.0 / preprocess_list_[0].img_max_value);
			cv::resize(image, image, cv::Size(preprocess_list_[0].img_width, preprocess_list_[0].img_height), 0, 0, cv::INTER_LINEAR);
			double maxValue;
			cv::minMaxLoc(image, NULL, &maxValue, NULL, NULL);
			cv::divide(image, cv::Scalar(maxValue, maxValue, maxValue), image);
			cv::subtract(image, cv::Scalar(preprocess_list_[0].img_mean[0], preprocess_list_[0].img_mean[1], preprocess_list_[0].img_mean[2]), image);
			cv::divide(image, cv::Scalar(preprocess_list_[0].img_std[0], preprocess_list_[0].img_std[1], preprocess_list_[0].img_std[2]), image);
		}
		catch (cv::Exception &e)
		{
			// output exception information
			LOG(ERROR) << "message: " << e.what();
			return YJH_AI_INPUT_IMGPREPROCESS_ERROR;
		}
		return DLSUCCESSED;
	}


	int BaseModelPredictor::Softmax(cv::Mat &tensor)
	{
		double max = 0.0;
		float sum = 0.0;
		try
		{
			cv::minMaxLoc(tensor, NULL, &max, NULL, NULL);
			cv::exp((tensor - max), tensor);
			sum = cv::sum(tensor)[0];
			tensor /= sum;
		}
		catch (cv::Exception &e)
		{
			// output exception information
			LOG(ERROR) << "message: " << e.what();
			return YJH_AI_OPENCV_SOFTMAX_ERROR;
		}
		return DLSUCCESSED;
	}

	int BaseModelPredictor::InitModel()
	{
		//not implemented
		return DLFAILED;
		
	}

	int BaseModelPredictor::Predict(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs)
	{
		if(auto_preprocess_.size() != 0)
		{
			std::vector<cv::Mat> imgs;
			imgs.emplace_back(img);
			for(unsigned int i=0;i<auto_preprocess_.size();i++)
			{
				std::shared_ptr<Transform> trnasformer = TransformFactory::getInstance().getClassByName(auto_preprocess_[i]);
				trnasformer->Apply(imgs,preprocess_list_);
			}
			return PredictImp(imgs[0],outputBlobs);
		}	
		else
		{
			return PredictImp(img,outputBlobs);
		}
	}


	int BaseModelPredictor::Predict(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs)
	{
		
		if(auto_preprocess_.size() != 0)
		{	
			std::vector<cv::Mat> img_list = imgs;		
			for(unsigned int i=0;i<auto_preprocess_.size();i++)
			{
				std::shared_ptr<Transform> trnasformer = TransformFactory::getInstance().getClassByName(auto_preprocess_[i]);
				trnasformer->Apply(img_list,preprocess_list_);
			}
			return PredictImp(img_list,outputBlobs);		
		}
		else
		{
			return PredictImp(imgs,outputBlobs);
		}
	}

	int BaseModelPredictor::BatchPredict(const std::vector<std::vector<cv::Mat>> &imgs, std::vector<cv::Mat> &outputBlobs)
	{
		
		if(auto_preprocess_.size() != 0)
		{	
			std::vector<std::vector<cv::Mat>> img_list = imgs;			
			for(unsigned int j=0;j<imgs.size();j++)
			{
				for(unsigned int i=0;i<auto_preprocess_.size();i++)
				{
					std::shared_ptr<Transform> trnasformer = TransformFactory::getInstance().getClassByName(auto_preprocess_[i]);
					trnasformer->Apply(img_list[j],preprocess_list_);
				}
			}					
			return BatchPredictImp(img_list,outputBlobs);
		}
		else
		{
			return BatchPredictImp(imgs,outputBlobs);
		}
		
	}
       

	int BaseModelPredictor::PredictImp(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs)
	{
		//not implemented
		LOG(ERROR) << "not implemented Predict(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs) function ";
		return DLFAILED;
	}

	int BaseModelPredictor::PredictImp(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs)
	{	
		//not implemented
		LOG(ERROR) << "not implemented Predict(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs) function ";
		return DLFAILED;
	}

	int BaseModelPredictor::BatchPredictImp(const std::vector<std::vector<cv::Mat>> &imgs, std::vector<cv::Mat> &outputBlobs)
	{
		//not implemented
		LOG(ERROR) << "not implemented BatchPredict(const std::vector<std::vector<cv::Mat>> &imgs, std::vector<cv::Mat> &outputBlobs) ";
		return DLFAILED;
	}

	void BaseModelPredictor::DeInitModel()
	{
		// LOG(ERROR) << "not implemented DeInitModel()";
		//not implemented		
	}

}