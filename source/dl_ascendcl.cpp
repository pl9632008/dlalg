#include "dl_ascendcl.h"
#include "dl_common.h"


// #include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>
#include <algorithm>

namespace yjh_deeplearning
{

	AscendCLPredictor::AscendCLPredictor(dlalg_jsons::ModelInfo &modelInfo):BaseModelPredictor(modelInfo)
	{
		
	}


#ifdef USE_ASCENDCL
	

	int AscendCLPredictor::GetTypeSizeFormACL(aclDataType type)
	{
		int size=0;
		switch (type)
        {
			case ACL_INT32: size = 4; break;
			case ACL_FLOAT: size=4; break;
			case ACL_FLOAT16: size=2; break;
			case ACL_INT8: size=1; break;
			case ACL_UINT8: size=1; break;
			case ACL_BOOL: size=1; break;
        }
		return size;
	}


	int AscendCLPredictor::GetOpencvTypeFormACL(aclDataType type)
	{
		int opencv_type=0;
		switch (type)
        {
			case ACL_INT32: opencv_type = CV_32S; break;
			case ACL_FLOAT: opencv_type=CV_32F; break;
			case ACL_FLOAT16: opencv_type=CV_16S; break;
			case ACL_INT8: opencv_type=CV_8S; break;
			case ACL_UINT8: opencv_type=CV_8U; break;
			case ACL_BOOL: opencv_type=CV_8U; break;
        }
		return opencv_type;
	}

#endif

	int AscendCLPredictor::InitModel()
	{
#ifdef USE_ASCENDCL
		if(gpu_index_ < 0)
		{
			LOG(ERROR) << "InitAscendCLModel error, need npu";
			return DLFAILED;
		}
		aclError ret = aclInit(nullptr);
		if (ret != ACL_SUCCESS)
		{
			LOG(ERROR) << "InitAscendCL lib error "<<ret;
			return DLFAILED;
		}
		ret = aclrtSetDevice(gpu_index_);
		if (ret != ACL_SUCCESS)
		{
			LOG(ERROR) << "set ascendcl device error "<<ret;
			return DLFAILED;
		}
		ret = aclmdlLoadFromFile(weight_path_.c_str(), &modelId_);
		if (ret != ACL_SUCCESS)
		{
			LOG(ERROR) << "load ascendcl weights error, "<<ret<<" path:"<<weight_path_;
			return DLFAILED;
		}

		model_desc_ = aclmdlCreateDesc();
  		ret = aclmdlGetDesc(model_desc_, modelId_);
		if (ret != ACL_SUCCESS)
		{
			LOG(ERROR) << "get ascendcl model decs error "<<ret;
			return DLFAILED;
		}
				
		input_data_set_ = aclmdlCreateDataset();
		n_inputs_ = aclmdlGetNumInputs(model_desc_);
		
		for(int i=0;i<n_inputs_;i++)
		{
			auto input_len = aclmdlGetInputSizeByIndex(model_desc_, i);			
			ret = aclrtMalloc(&input_device_buffer_[i], input_len, ACL_MEM_MALLOC_HUGE_FIRST);
			if (ret != ACL_SUCCESS)
			{
				LOG(ERROR) << "aclrtMalloc error "<<ret;
				return DLFAILED;
			}
			input_data_buffer_[i] = aclCreateDataBuffer(input_device_buffer_[i], input_len);
			ret = aclmdlAddDatasetBuffer(input_data_set_, input_data_buffer_[i]);
			if (ret != ACL_SUCCESS)
			{
				LOG(ERROR) << "input aclmdlAddDatasetBuffer error "<<ret;
				return DLFAILED;
			}
			auto data_type = aclmdlGetInputDataType(model_desc_, i);
			input_type_size_[i] = GetTypeSizeFormACL(data_type);			
		}
		
		output_data_set_ = aclmdlCreateDataset();
		n_outputs_ = aclmdlGetNumOutputs(model_desc_);
		
		for(int i=0;i<n_outputs_;i++)
		{
			auto output_len = aclmdlGetOutputSizeByIndex(model_desc_, i);
			output_data_size_[i] = output_len;
			ret = aclrtMalloc(&output_device_buffer_[i], output_len, ACL_MEM_MALLOC_HUGE_FIRST);
			if (ret != ACL_SUCCESS)
			{
				LOG(ERROR) << "aclrtMalloc error "<<ret;
				return DLFAILED;
			}
			output_data_buffer_[i] = aclCreateDataBuffer(output_device_buffer_[i], output_len);
			ret = aclmdlAddDatasetBuffer(output_data_set_, output_data_buffer_[i]);
			if (ret != ACL_SUCCESS)
			{
				LOG(ERROR) << "output aclmdlAddDatasetBuffer error "<<ret;
				return DLFAILED;
			}
			ret = aclmdlGetOutputDims(model_desc_, i, &output_dims_[i]);
			if (ret != ACL_SUCCESS)
			{
				LOG(ERROR) << "output aclmdlGetCurOutputDims error "<<ret;
				return DLFAILED;
			}
			auto data_type = aclmdlGetOutputDataType(model_desc_, i);
			output_opencv_type_[i] = GetOpencvTypeFormACL(data_type);

		}

		return DLSUCCESSED;

#endif
		LOG(ERROR) << "not open MACRO USE_ASCENDCL";
		return DLFAILED;
	}


	int AscendCLPredictor::PredictImp(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs)
	{
		return Predict(std::vector<cv::Mat>{img},outputBlobs);
	}

	int AscendCLPredictor::PredictImp(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs)
	{
#ifdef USE_ASCENDCL
		aclError ret;
		if(imgs.size() != n_inputs_)
		{
			LOG(ERROR) << "input img size error: ninputs is "<<n_inputs_<<" but img size is: "<<imgs.size();
			return DLFAILED;
		}
		std::vector<cv::Mat> mat_channels;
		for(unsigned int i=0;i<imgs.size();i++)
		{
			cv::split(imgs[i], mat_channels);
			int copy_size = imgs[i].cols * imgs[i].rows * input_type_size_[i];		
			for (unsigned int k = 0; k < mat_channels.size(); ++k)
			{			
				ret = aclrtMemcpy((unsigned char *)input_device_buffer_[i]+ k* copy_size, copy_size, mat_channels.at(k).data, copy_size, ACL_MEMCPY_HOST_TO_DEVICE);			
				if(ret != ACL_SUCCESS)
				{
					LOG(ERROR) << "aclrtMemcpy error ";
					return DLFAILED;
				}				
			}	
		}
		ret = aclmdlExecute(modelId_, input_data_set_, output_data_set_);
		if(ret != ACL_SUCCESS)
		{
			LOG(ERROR) << "aclmdlExecute error ";
			return DLFAILED;
		}
		outputBlobs.clear();
		for(unsigned int i=0;i<n_outputs_;i++)
		{
			std::vector<int> out_size(output_dims_[i].dimCount);		
			for (int j = 0; j < output_dims_[i].dimCount; j++)
			{
				out_size[j] = output_dims_[i].dims[j];				
			}
			
			cv::Mat outputBlob(out_size, output_opencv_type_[i]);	
			ret = aclrtMemcpy(outputBlob.data, output_data_size_[i], output_device_buffer_[i], output_data_size_[i], ACL_MEMCPY_DEVICE_TO_HOST);
			if(ret != ACL_SUCCESS)
			{
				LOG(ERROR) << "aclrtMemcpy error ";
				return DLFAILED;
			}
			outputBlobs.emplace_back(outputBlob);
		}
		return DLSUCCESSED;

#endif
		LOG(ERROR) << "not open MACRO USE_ASCENDCL";
		return DLFAILED;
	}



	void AscendCLPredictor::DeInitModel()
	{
#ifdef USE_ASCENDCL
		aclmdlDestroyDesc(model_desc_);
		aclmdlUnload(modelId_);
		for(unsigned int i=0;i<MAX_MEMORY_BLOCK_SIZE;i++)
		{
			if(input_device_buffer_[i] != nullptr)
			{
				aclrtFree(input_device_buffer_[i]);
				input_device_buffer_[i] = nullptr;
			}
			if(input_data_buffer_[i] != nullptr)
			{
				aclDestroyDataBuffer(input_data_buffer_[i]);
				input_data_buffer_[i] = nullptr;
			}			
		}
		aclmdlDestroyDataset(input_data_set_);
		input_data_set_ = nullptr;
		for(unsigned int i=0;i<MAX_MEMORY_BLOCK_SIZE;i++)
		{
			if(output_device_buffer_[i] != nullptr)
			{
				aclrtFree(output_device_buffer_[i]);
				output_device_buffer_[i] = nullptr;
			}
			if(output_data_buffer_[i] != nullptr)
			{
				aclDestroyDataBuffer(output_data_buffer_[i]);
				output_data_buffer_[i] = nullptr;
			}
		}
		aclmdlDestroyDataset(output_data_set_);
		output_data_set_ = nullptr;

#endif			
	}

}


