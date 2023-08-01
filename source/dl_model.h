#ifndef DL_MODEL_H
#define DL_MODEL_H

#include "dl_format.h"
#include "dl_common.h"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>


namespace yjh_deeplearning
{
       

        /**
         * @brief 模型推理通用类，支持oencv和libtorch框架，如果通用处理函数接口不能满足需求，用户可以继承，在子类添加自己所需功能。
         
         */
        class BaseModelPredictor
        {
        public:
                BaseModelPredictor(dlalg_jsons::ModelInfo &modelInfo);
                ~BaseModelPredictor() = default;

                //通用前处理函数，子类可以直接使用，也可以自己去实现
                int GeneralPreProcess(cv::Mat &image);
          
                int PreProcessByMulMaxValue(cv::Mat &image);
               
                int Softmax(cv::Mat &tensor);

                // 初始化函数，子类必须实现
                virtual int InitModel()=0;
                // 单张量推理函数，单张量输入
                int Predict(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs);
                // 多张量推理函数，多张量输入
                int Predict(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs);
                // 批推理函数
                int BatchPredict(const std::vector<std::vector<cv::Mat>> &imgs, std::vector<cv::Mat> &outputBlobs);
                // 资源释放函数

                 // 单张量推理函数，单张量输入
                virtual int PredictImp(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs);
                // 多张量推理函数，多张量输入
                virtual int PredictImp(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs);
                // 批推理函数
                virtual int BatchPredictImp(const std::vector<std::vector<cv::Mat>> &imgs, std::vector<cv::Mat> &outputBlobs);
                // 资源释放函数
                
                virtual void DeInitModel();
         
        protected:
                std::vector<std::string> auto_preprocess_;	
                std::vector<dlalg_jsons::PreprocessInfo> preprocess_list_;

                bool is_dynamic_infer{false};
                
                unsigned int batch_size_;
                std::string weight_path_;
                std::string cfg_path_;
                int gpu_index_;

                std::vector<std::string> name_input_;
                std::vector<std::string> name_output_;

                std::string model_type_;              

                bool is_half_;

        };

}

#endif