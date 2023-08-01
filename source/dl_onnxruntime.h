#ifndef DL_ONNXRUNTIMEH
#define DL_ONNXRUNTIMEH

#include "dl_format.h"
#include "dl_common.h"
#include "dl_model.h"


#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>


namespace yjh_deeplearning
{
       
        class  ONNXRuntimePredictor:public BaseModelPredictor
        {
        public:
                ONNXRuntimePredictor(dlalg_jsons::ModelInfo &modelInfo);
                ~ONNXRuntimePredictor() = default;
             
                int InitModel();
                int PredictImp(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs);                
                int PredictImp(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs);

        private:

                std::string onnxruntime_customop_library_path_;
                int onnxruntime_intraOpNum_;
               
                std::vector<const char*> input_node_names_{};
                std::vector<const char*> output_node_names_{};
                std::vector<float> intput_tensor_value_;
           
#ifdef USE_ONNXRUNTIME

                int GetOpencvTypeFormORT(ONNXTensorElementDataType type);
                int GetTypeSizeFormORT(ONNXTensorElementDataType type);
                Ort::AllocatorWithDefaultOptions allocator_;
                Ort::MemoryInfo MemoryInfo();
                Ort::Env ort_env_;
                Ort::Session onnx_session_{nullptr};
                std::vector<Ort::Value> ort_inputs_;
                                               
#endif
        };

}

#endif