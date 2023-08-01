#include "dl_onnxruntime.h"
#include "dl_common.h"


#include <glog/logging.h>
#include <algorithm>

namespace yjh_deeplearning
{

	ONNXRuntimePredictor::ONNXRuntimePredictor(dlalg_jsons::ModelInfo &modelInfo):BaseModelPredictor(modelInfo)
	{
		
		onnxruntime_customop_library_path_ = modelInfo.onnxruntime_customop_library;
		onnxruntime_intraOpNum_ = modelInfo.onnxruntime_intraOpNum;
		
	}
	

	int ONNXRuntimePredictor::InitModel()
	{	
#ifdef USE_ONNXRUNTIME
		try{
			ort_env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "inference");
			Ort::SessionOptions session_options;
			if(onnxruntime_customop_library_path_ != "" && onnxruntime_customop_library_path_.empty())
			{
				void* handle = nullptr;				
				Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary((OrtSessionOptions*)session_options, onnxruntime_customop_library_path_.data(), &handle));
			}
#ifdef USE_ONNXRUNTIME_CUDA
			if(gpu_index_ >= 0)
			{
				OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, gpu_index_);		
			}
		
#endif	
			session_options.SetIntraOpNumThreads(onnxruntime_intraOpNum_);			
			session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
			onnx_session_ = Ort::Session(ort_env_, weight_path_.data(), session_options);

			input_node_names_.resize(onnx_session_.GetInputCount());			
			for(unsigned int i=0;i<onnx_session_.GetInputCount();i++)
			{			
				input_node_names_[i] = onnx_session_.GetInputName(i, allocator_);				
			}

			output_node_names_.resize(onnx_session_.GetOutputCount());
			for(unsigned int i=0;i<onnx_session_.GetOutputCount();i++)
			{			
				output_node_names_[i] = onnx_session_.GetOutputName(i, allocator_);
			}

		}
		catch(const Ort::Exception &exception)
		{
			LOG(ERROR) << "load onnxruntime library error "<<exception.what();
			return DLFAILED;
		}
		if(input_node_names_.size() != preprocess_list_.size())
		{
			LOG(ERROR) << "preprocess_list_ size error its size = "<<preprocess_list_.size()<<" input_node_names_.size() = "<<input_node_names_.size();
			return DLFAILED;
		}
	
		return DLSUCCESSED;
#endif
		LOG(ERROR) << "not open MACRO USE_ONNXRUNTIME";
		return DLFAILED;
	}

#ifdef USE_ONNXRUNTIME
	int ONNXRuntimePredictor::GetOpencvTypeFormORT(ONNXTensorElementDataType type)
	{
		int opencv_type=0;
		switch (type)
        {
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: opencv_type = CV_32S; break;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: opencv_type=CV_32F; break;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: opencv_type=CV_16S; break;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: opencv_type=CV_8S; break;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: opencv_type=CV_8U; break;
        }
		return opencv_type;
	}

	int ONNXRuntimePredictor::GetTypeSizeFormORT(ONNXTensorElementDataType type)
	{
		int size=0;
		switch (type)
        {
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: size = 4; break;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: size=4; break;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: size=2; break;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: size=1; break;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: size=1; break;
        }
		return size;
	}
#endif

	int ONNXRuntimePredictor::PredictImp(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs)
	{
		return Predict(std::vector<cv::Mat>{img},outputBlobs);		
	}

	int ONNXRuntimePredictor::PredictImp(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs)
	{
		
#ifdef USE_ONNXRUNTIME

		if(imgs.size() != input_node_names_.size())
		{
			LOG(ERROR) << "input mat num error!";
			return YJH_AI_OPENCV_INFERENCE_ERROR;
		}
		std::vector<cv::Mat> mat_channels;
		ort_inputs_.clear();
		intput_tensor_value_.clear();
		for (unsigned int i = 0; i < imgs.size(); i++)
		{	
			cv::split(imgs[i], mat_channels);
			
			intput_tensor_value_.resize(mat_channels.size()*imgs[i].rows*imgs[i].cols);
			for (unsigned int k = 0; k < mat_channels.size(); ++k)
			{					
				std::memcpy(intput_tensor_value_.data() + k *( mat_channels[k].rows * mat_channels[k].cols),mat_channels.at(k).data, imgs[i].rows * imgs[i].cols * sizeof(float));				
			}
			std::vector<int64_t> input_mask_node_dims = {1,mat_channels.size(), imgs[i].rows, imgs[i].cols};
			try{
				Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);			
				Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info_handler, intput_tensor_value_.data(), intput_tensor_value_.size(), input_mask_node_dims.data(), input_mask_node_dims.size());
				ort_inputs_.emplace_back(std::move(input_tensor));
			}
			catch(const Ort::Exception &exception)
			{
				LOG(ERROR) << "load img to ort error"<<exception.what();
				return DLFAILED;
			}
		}
		
		try{
			auto output_tensors = onnx_session_.Run(Ort::RunOptions{nullptr}, input_node_names_.data(), ort_inputs_.data(), ort_inputs_.size(), output_node_names_.data(), onnx_session_.GetOutputCount());
			std::vector<int64_t> shape;
			outputBlobs.clear(); 
			for (int i = 0; i <onnx_session_.GetOutputCount(); ++i) {
				Ort::TensorTypeAndShapeInfo Info = output_tensors[i].GetTensorTypeAndShapeInfo();
				
				shape = Info.GetShape();
				std::vector<int> cv_shape(shape.begin(),shape.end());
				cv::Mat outputBlob(cv_shape, GetOpencvTypeFormORT(onnx_session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType()));
				Ort::Value &pred = output_tensors[i];
				int shape_size=1;
				for (int j = 0; j < cv_shape.size(); ++j) {
					shape_size *= cv_shape[j];
				}
				std::memcpy(outputBlob.data,pred.GetTensorData<void>(),shape_size*GetTypeSizeFormORT(onnx_session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType()));
				
				outputBlobs.emplace_back(outputBlob);
				}
		}
		catch(const Ort::Exception &exception)
		{
			LOG(ERROR) << "inference error: "<<exception.what();
			return DLFAILED;
		}

		return DLSUCCESSED;
#endif
		LOG(ERROR) << "not open MACRO USE_ONNXRUNTIME";
		return DLFAILED;
	}


}