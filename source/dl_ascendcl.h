#ifndef DL_ASCENDCL_H
#define DL_ASCENDCL_H

#include "dl_format.h"
#include "dl_common.h"
#include "dl_model.h"



#ifdef USE_ASCENDCL
#include "acl/acl.h"
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>


namespace yjh_deeplearning
{

       
        class AscendCLPredictor:public BaseModelPredictor
        {
        public:
                AscendCLPredictor(dlalg_jsons::ModelInfo &modelInfo);
                ~AscendCLPredictor() = default;

                
                // 华为npu模型推理
                int InitModel();
                int PredictImp(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs);
                int PredictImp(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs);
                void DeInitModel();
              
        private:          
         
#ifdef USE_ASCENDCL
                // tensorrt
                int GetTypeSizeFormACL(aclDataType type);
                int GetOpencvTypeFormACL(aclDataType type);
                
                aclmdlDesc *model_desc_;

                aclmdlDataset *input_data_set_;
                aclDataBuffer *input_data_buffer_[MAX_MEMORY_BLOCK_SIZE]{nullptr};
                void* input_device_buffer_[MAX_MEMORY_BLOCK_SIZE]{nullptr};
                int n_inputs_;
                int input_type_size_[MAX_MEMORY_BLOCK_SIZE]{0};


                aclmdlDataset *output_data_set_;
                aclDataBuffer *output_data_buffer_[MAX_MEMORY_BLOCK_SIZE]{nullptr};
                void* output_device_buffer_[MAX_MEMORY_BLOCK_SIZE]{nullptr};
                int n_outputs_;


                aclmdlIODims output_dims_[MAX_MEMORY_BLOCK_SIZE];
                int output_opencv_type_[MAX_MEMORY_BLOCK_SIZE]{0};
                int output_data_size_[MAX_MEMORY_BLOCK_SIZE]{0};
              
               
                uint32_t modelId_;
#endif


        };

}

#endif