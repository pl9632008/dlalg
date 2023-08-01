#include "dl_libtorch.h"
#include "dl_common.h"


#include <glog/logging.h>
#include <algorithm>

namespace yjh_deeplearning
{

	LibTorchPredictor::LibTorchPredictor(dlalg_jsons::ModelInfo &modelInfo):BaseModelPredictor(modelInfo)
	{
	
	}


	int LibTorchPredictor::InitModel()
	{
		if(preprocess_list_.size() == 0)
		{
			LOG(ERROR) << "preprocess config error";
			return YJH_AI_INPUT_STDORMEAN_ERROR;
		}
#ifdef USE_LIBTORCH
		try
		{
			torch_net_ = torch::jit::load(weight_path_);
			torch_net_.eval();
			for(unsigned int i=0;i<preprocess_list_.size();i++)
			{			
				if (gpu_index_ >= 0)
				{

						torch_net_.to(torch::Device(torch::kCUDA, gpu_index_)); // 模型放到GPU上推理
						torch_tensor_ = torch::randn({1, preprocess_list_[i].img_channel, preprocess_list_[i].img_height, preprocess_list_[i].img_width}).cuda();
						if (is_half_ == true)
						{
							torch_net_.to(torch::kHalf); // 模型FP32转FP16
							
							torch_tensor_.to(torch::kHalf);
						}
						
				}
				else
				{
					torch_tensor_ = torch::randn({1, preprocess_list_[i].img_channel, preprocess_list_[i].img_height, preprocess_list_[i].img_width});
				}
				input_tensors_.emplace_back(torch_tensor_);
			}
			torch_net_.forward(input_tensors_);
		}
		catch (const c10::Error &e)
		{
			LOG(ERROR) << "error loading libtorch the model";
			return YJH_AI_LIBTORCH_INIT_ERROR;
		}
		return DLSUCCESSED;
#endif
		LOG(ERROR) << "not open MACRO USE_LIBTORCH";
		return DLFAILED;
	}



	int LibTorchPredictor::PredictImp(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs)
	{
#ifdef USE_LIBTORCH		
		outputBlobs.clear();
		try
		{	
			if (gpu_index_ >= 0)
			{
				torch_tensor_ = torch::from_blob(img.data, {1, img.rows, img.cols, img.channels()}, torch::kFloat).to(torch::Device(torch::kCUDA, gpu_index_));
				torch_tensor_ = torch_tensor_.permute({0, 3, 1, 2});
				if (is_half_ == true)
				{
					torch_tensor_.to(torch::kHalf); // 将模型由FP32转换成FP16,提升速度,cpu不支持
				}
			}
			else
			{
				torch_tensor_ = torch::from_blob(img.data, {1, img.rows, img.cols, img.channels()}, torch::kFloat);
				torch_tensor_ = torch_tensor_.permute({0, 3, 1, 2});
			}
			auto result_tensor = torch_net_.forward({torch_tensor_});
			std::vector<c10::IValue> result;
			unsigned int result_size = 0;
			if(result_tensor.isTuple())
			{
				result = result_tensor.toTuple()->elements();
				result_size = result.size();
			}
			else if(result_tensor.isTensor())
			{
				result_size = 1;
			}
			else
			{
				LOG(ERROR) << "error inference libtorch, return error result type";
				return YJH_AI_LIBTORCH_INFERENCE_ERROR;
			}
			torch::Tensor outTensor;
			for (unsigned int i = 0; i < result_size; i++)
			{	
				if(result_tensor.isTuple())
				{
					outTensor = result[i].toTensor();
				}
				else
				{
					outTensor = result_tensor.toTensor();
				}				
				auto sizes = outTensor.sizes();
				std::vector<int> out_size(outTensor.dim());
				for (int i = 0; i < outTensor.dim(); i++)
				{
					out_size[i] = sizes[i];
				}
				cv::Mat outputBlob(out_size, CV_32F);
				std::memcpy(outputBlob.data, outTensor.data_ptr(), outTensor.numel() * sizeof(torch::kFloat));
				outputBlobs.emplace_back(outputBlob);
			}	
		}
		catch (const c10::Error &e)
		{
			LOG(ERROR) << "error inference libtorch the model";
			return YJH_AI_LIBTORCH_INFERENCE_ERROR;
		}

		return DLSUCCESSED;
#endif
		LOG(ERROR) << "not open MACRO USE_LIBTORCH";
		return DLFAILED;
	}

int LibTorchPredictor::PredictImp(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs)
	{
#ifdef USE_LIBTORCH
		if(imgs.size() != preprocess_list_.size())
		{
			LOG(ERROR) << "error inference ,input num is error";
			return YJH_AI_LIBTORCH_INFERENCE_ERROR;
		}
		outputBlobs.clear();
		try
		{	
			input_tensors_.clear();
			for(unsigned int i=0;i<imgs.size();i++)
			{
				if (gpu_index_ >= 0)
				{
					torch_tensor_ = torch::from_blob(imgs[i].data, {1, imgs[i].rows, imgs[i].cols, imgs[i].channels()}, torch::kFloat).to(torch::Device(torch::kCUDA, gpu_index_));
					torch_tensor_ = torch_tensor_.permute({0, 3, 1, 2});
					if (is_half_ == true)
					{
						torch_tensor_.to(torch::kHalf); // 将模型由FP32转换成FP16,提升速度,cpu不支持
					}
				}
				else
				{
					torch_tensor_ = torch::from_blob(imgs[i].data, {1, imgs[i].rows, imgs[i].cols, imgs[i].channels()}, torch::kFloat);
					torch_tensor_ = torch_tensor_.permute({0, 3, 1, 2});
				}
				input_tensors_.emplace_back(torch_tensor_);
			}
			auto result_tensor = torch_net_.forward({torch_tensor_});
			std::vector<c10::IValue> result;
			unsigned int result_size = 0;
			if(result_tensor.isTuple())
			{
				result = result_tensor.toTuple()->elements();
				result_size = result.size();
			}
			else if(result_tensor.isTensor())
			{
				result_size = 1;
			}
			else
			{
				LOG(ERROR) << "error inference libtorch, return error result type";
				return YJH_AI_LIBTORCH_INFERENCE_ERROR;
			}
			torch::Tensor outTensor;
			for (unsigned int i = 0; i < result_size; i++)
			{	
				if(result_tensor.isTuple())
				{
					outTensor = result[i].toTensor();
				}
				else
				{
					outTensor = result_tensor.toTensor();
				}				
				auto sizes = outTensor.sizes();
				std::vector<int> out_size(outTensor.dim());
				for (int i = 0; i < outTensor.dim(); i++)
				{
					out_size[i] = sizes[i];
				}
				cv::Mat outputBlob(out_size, CV_32F);
				std::memcpy(outputBlob.data, outTensor.data_ptr(), outTensor.numel() * sizeof(torch::kFloat));
				outputBlobs.emplace_back(outputBlob);
			}	
		}
		catch (const c10::Error &e)
		{
			LOG(ERROR) << "error inference libtorch the model";
			return YJH_AI_LIBTORCH_INFERENCE_ERROR;
		}

		return DLSUCCESSED;
#endif
		LOG(ERROR) << "not open MACRO USE_LIBTORCH";
		return DLFAILED;
	}




}