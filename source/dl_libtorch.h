#ifndef DL_LIBTORCH_H
#define DL_LIBTORCH_H

#include "dl_format.h"
#include "dl_common.h"
#include "dl_model.h"

#ifdef USE_LIBTORCH
#include "torch/torch.h"
#include "torch/script.h"
#endif


#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>


namespace yjh_deeplearning
{

        
        class LibTorchPredictor:public BaseModelPredictor
        {
        public:
                LibTorchPredictor(dlalg_jsons::ModelInfo &modelInfo);
                ~LibTorchPredictor() = default;

                               
                int InitModel();         
                int PredictImp(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs);                
                int PredictImp(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs);

              
       private:
      
#ifdef USE_LIBTORCH
                torch::jit::script::Module torch_net_; // torch推理网络
                std::vector<c10::IValue> input_tensors_;           // tensor格式图片
                torch::Tensor torch_tensor_;
#endif

        };

}

#endif