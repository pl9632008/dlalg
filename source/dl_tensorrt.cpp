#include "dl_tensorrt.h"
#include "dl_common.h"

#ifdef USE_TENSORRT
#include "NvInferPlugin.h"
#endif


// #include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>
#include <algorithm>

namespace yjh_deeplearning
{

	TenorrtPredictor::TenorrtPredictor(dlalg_jsons::ModelInfo &modelInfo):BaseModelPredictor(modelInfo)
	{
		
	}


#ifdef USE_TENSORRT
	class TRTLogger : public nvinfer1::ILogger
	{
	public:
		void log(Severity severity, const char *msg) noexcept override
		{
			switch (severity)
			{
			case Severity::kINFO:
				// MMDEPLOY_INFO("TRTNet: {}", msg);
				break;
			case Severity::kWARNING:
				LOG(WARNING) << "TRTNet:" << msg;
				break;
			case Severity::kERROR:
			case Severity::kINTERNAL_ERROR:
				LOG(ERROR) << "TRTNet:" << msg;
				break;
			default:
				break;
			}
		}
		static TRTLogger &get()
		{
			static TRTLogger trt_logger{};
			return trt_logger;
		}
	};

	int TenorrtPredictor::GetTypeSizeFormRT(nvinfer1::DataType type)
	{
		int size=0;
		switch (type)
        {
			case nvinfer1::DataType::kINT32: size = 4; break;
			case nvinfer1::DataType::kFLOAT: size=4; break;
			case nvinfer1::DataType::kHALF: size=2; break;
			case nvinfer1::DataType::kINT8: size=1; break;
			case nvinfer1::DataType::kBOOL: size=1; break;
        }
		return size;
	}

	int TenorrtPredictor::GetOpencvTypeFormRT(nvinfer1::DataType type)
	{
		int opencv_type=0;
		switch (type)
        {
			case nvinfer1::DataType::kINT32: opencv_type = CV_32S; break;
			case nvinfer1::DataType::kFLOAT: opencv_type=CV_32F; break;
			case nvinfer1::DataType::kHALF: opencv_type=CV_16S; break;
			case nvinfer1::DataType::kINT8: opencv_type=CV_8S; break;
			case nvinfer1::DataType::kBOOL: opencv_type=CV_8U; break;
        }
		return opencv_type;
	}

#endif

	int TenorrtPredictor::InitModel()
	{
#ifdef USE_TENSORRT
		if(gpu_index_ < 0)
		{
			LOG(ERROR) << "InitTensorrtModel error, need gpu";
			return DLFAILED;
		}
		if(is_dynamic_infer == true)
		{
			LOG(ERROR) << "InitTensorrtModel error, can not support dynamic inference";
			return DLFAILED;
		}
		char *trtModelStream{nullptr};
		size_t size{0};
		std::ifstream file(weight_path_, std::ios::binary);
		if (file.good())
		{
			file.seekg(0, file.end);
			size = file.tellg();
			file.seekg(0, file.beg);
			trtModelStream = new char[size];
			assert(trtModelStream);
			file.read(trtModelStream, size);
			file.close();
		}
		else
		{
			LOG(ERROR) << "InitTensorrtModel error, can not open " << weight_path_;
			return DLFAILED;
		}

		initLibNvInferPlugins(&TRTLogger::get(), "");
    	cudaSetDevice(gpu_index_);

		nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(TRTLogger::get());
		if (runtime == nullptr)
		{
			LOG(ERROR) << "InitTensorrtModel error, can not create infer runtime";
			return DLFAILED;
		}
		engine_.reset(runtime->deserializeCudaEngine(trtModelStream, size));
		if (engine_ == nullptr)
		{
			LOG(ERROR) << "InitTensorrtModel error, can not create CudaEngine";
			return DLFAILED;
		}
		context_.reset(engine_->createExecutionContext());
		if (context_ == nullptr)
		{
			LOG(ERROR) << "InitTensorrtModel error, can not create context";
			return DLFAILED;
		}
		delete[] trtModelStream;
		auto n_bindings = engine_->getNbBindings();
		for (int i = 0; i < n_bindings; ++i) {
			auto dims = engine_->getBindingDimensions(i);
			auto data_type = engine_->getBindingDataType(i);		
			binding_opencv_type_[i]= GetOpencvTypeFormRT(data_type);
			binding_type_size_[i] = GetTypeSizeFormRT(data_type);
			if (engine_->bindingIsInput(i)) 
			{			
				if(preprocess_list_.size()<(input_num_+1))
				{
					LOG(ERROR) << "InitTensorrtModel error, config file error";
					return DLFAILED;
				}
				if(cudaMalloc(&bindings_[i], batch_size_*preprocess_list_[input_num_].img_width*preprocess_list_[input_num_].img_height*preprocess_list_[input_num_].img_channel * GetTypeSizeFormRT(data_type))!=0)
				{
					LOG(ERROR) << "InitTensorrtModel error, cudaMalloc error";
					return DLFAILED;
				}
				input_index_[input_num_] = i;
				input_num_++;
			}
			else 
			{
				auto output_size = 1;
				nvinfer1::Dims dim;	
				dim.d[0] = batch_size_;	
				for(int j=1;j<dims.nbDims;j++)
				{
					output_size *= dims.d[j];
					dim.d[j] = dims.d[j];																
				}
				dim.nbDims = dims.nbDims;
				if(cudaMalloc(&bindings_[i], batch_size_* output_size * GetTypeSizeFormRT(data_type))!=0)
				{
					LOG(ERROR) << "InitTensorrtModel error, cudaMalloc error";
					return DLFAILED;
				}			
				output_dims[output_num_] = dim;
				out_index_[output_num_] = i;
				output_num_++;
			}
		}
		if(cudaStreamCreate(&stream_)!=0)
		{
			LOG(ERROR) << "InitTensorrtModel error, cudaStreamCreate error";
			return DLFAILED;
		}
		for(unsigned int i=0;i<input_num_;i++)
		{		
			if(!context_->setBindingDimensions(input_index_[i], nvinfer1::Dims4(batch_size_, preprocess_list_[i].img_channel,preprocess_list_[i].img_height, preprocess_list_[i].img_width)))
			{
				LOG(ERROR) <<"set binding error";  
				return DLFAILED;
			}
		}
		if(!context_->enqueueV2(bindings_, stream_, nullptr))
		{
			LOG(ERROR) <<"forward error";  
			return DLFAILED;
		}
		return DLSUCCESSED;

#endif
		LOG(ERROR) << "not open MACRO USE_TENSORRT";
		return DLFAILED;
	}

int TenorrtPredictor::PredictImp(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs)
{
	return BatchPredictImp({{img}},outputBlobs);
}              

int TenorrtPredictor::PredictImp(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs)
{		
	return BatchPredictImp({imgs},outputBlobs);
}

	int TenorrtPredictor::BatchPredictImp(const std::vector<std::vector<cv::Mat>> &imgs, std::vector<cv::Mat> &outputBlobs)
	{
#ifdef USE_TENSORRT

		if(imgs.size()>batch_size_)
		{
			LOG(ERROR) << "input batch size greater than config";
			return DLFAILED;
		}
		for(unsigned int i=0;i<input_num_;i++)
		{
			binding_input_size_[i] = 0;
		}
		std::vector<cv::Mat> mat_channels;
		for(unsigned int i =0;i<imgs.size();i++)
		{
			if(imgs[i].size() != input_num_)
			{
				LOG(ERROR) << "input tensor nums not equal model input num";
				return DLFAILED;
			}
			for(unsigned int j=0;j<input_num_;j++)
			{				
				cv::split(imgs[i][j], mat_channels);
				if(mat_channels.size() != preprocess_list_[j].img_channel)
				{
					LOG(ERROR) << "input img channel not equal model channel ";
					return DLFAILED;
				}
				for (unsigned int k = 0; k < mat_channels.size(); ++k)
				{					
					if(cudaMemcpyAsync((unsigned char *)bindings_[input_index_[j]]+binding_input_size_[j], mat_channels.at(k).data, preprocess_list_[j].img_width*preprocess_list_[j].img_height * binding_type_size_[input_index_[j]], cudaMemcpyHostToDevice, stream_) != 0)
					{
						LOG(ERROR) << "cudaMemcpyAsync error ";
						return DLFAILED;
					}
					binding_input_size_[j]+=preprocess_list_[j].img_width*preprocess_list_[j].img_height * binding_type_size_[input_index_[j]];
				}					
			}
		}
		for(unsigned int i=0;i<input_num_;i++)
		{		
			if(!context_->setBindingDimensions(input_index_[i], nvinfer1::Dims4(imgs.size(), preprocess_list_[i].img_channel,preprocess_list_[i].img_height, preprocess_list_[i].img_width)))
			{
				LOG(ERROR) <<"set binding error";  
				return DLFAILED;
			}
		}
		if(!context_->enqueueV2(bindings_, stream_, nullptr))
		{
			LOG(ERROR) <<"forward error";  
			return DLFAILED;
		}
			
		outputBlobs.clear();
		for(unsigned int i=0;i<output_num_;i++)
		{
			std::vector<int> out_size(output_dims[i].nbDims);
			out_size[0] = imgs.size();
			// LOG(ERROR) <<"out_size[0] "<<out_size[0];  
			auto cpy_size = imgs.size();
			for (int j = 1; j < output_dims[i].nbDims; j++)
			{
				out_size[j] = output_dims[i].d[j];
				cpy_size *= output_dims[i].d[j];
				//  LOG(ERROR) <<"out_size[j] "<<out_size[j];  
			}
			cv::Mat outputBlob(out_size, binding_opencv_type_[out_index_[i]]);	
			// LOG(ERROR) <<"cudaMemcpyAsync error "<<out_index_[i]<<"|"<<cpy_size;  
			if(cudaMemcpyAsync(outputBlob.data, bindings_[out_index_[i]], cpy_size* binding_type_size_[out_index_[i]], cudaMemcpyDeviceToHost,stream_) != 0)
			{
				LOG(ERROR) <<"cudaMemcpyAsync error";  
				return DLFAILED;
			}			
			outputBlobs.emplace_back(outputBlob);
		}
		return DLSUCCESSED;
#endif
		LOG(ERROR) << "not open MACRO USE_TENSORRT";
		return DLFAILED;
	}


	void TenorrtPredictor::DeInitModel()
	{
#ifdef USE_TENSORRT
		for(unsigned int i=0;i<MAX_MEMORY_BLOCK_SIZE;i++)
		{
			if(bindings_[i] != nullptr)
			{
				cudaFree(bindings_[i]);
				bindings_[i] = nullptr;
			}
		}
		cudaStreamSynchronize(stream_);
		cudaStreamDestroy(stream_);		
#endif			
	}



}