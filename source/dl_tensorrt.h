#ifndef DL_TENSORRT_H
#define DL_TENSORRT_H

#include "dl_format.h"
#include "dl_common.h"
#include "dl_model.h"



#ifdef USE_TENSORRT
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>


namespace yjh_deeplearning
{

       
        class TenorrtPredictor:public BaseModelPredictor
        {
        public:
                TenorrtPredictor(dlalg_jsons::ModelInfo &modelInfo);
                ~TenorrtPredictor() = default;

                
                // tensorrt模型推理
                int InitModel();
                int PredictImp(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs);                
                int PredictImp(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs);
                int BatchPredictImp(const std::vector<std::vector<cv::Mat>> &imgs, std::vector<cv::Mat> &outputBlobs);
                void DeInitModel();
              
        private:          
         
#ifdef USE_TENSORRT
                // tensorrt
                int GetTypeSizeFormRT(nvinfer1::DataType type);
                int GetOpencvTypeFormRT(nvinfer1::DataType type);
                std::unique_ptr<nvinfer1::ICudaEngine> engine_;
                std::unique_ptr<nvinfer1::IExecutionContext> context_;
                cudaStream_t stream_;
                void* bindings_[MAX_MEMORY_BLOCK_SIZE]{nullptr};
                int binding_type_size_[MAX_MEMORY_BLOCK_SIZE]{0};
                int binding_opencv_type_[MAX_MEMORY_BLOCK_SIZE]{0};

                unsigned int input_num_{0};                
                std::map<unsigned int,unsigned int> input_index_{};
                std::map<unsigned int,unsigned int> binding_input_size_;
                
                unsigned int output_num_{0};
                std::map<unsigned int,unsigned int> out_index_{};
                std::map<unsigned int,nvinfer1::Dims> output_dims;                
               
                std::vector<cv::Mat> mat_channels_;
#endif


        };

}

#endif