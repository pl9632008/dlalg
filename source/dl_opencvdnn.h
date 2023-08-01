#ifndef DL_OPENCVDNN_H
#define DL_OPENCVDNN_H

#include "dl_format.h"
#include "dl_common.h"
#include "dl_model.h"


#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>


namespace yjh_deeplearning
{

       
        class OpencvDNNPredictor:public BaseModelPredictor
        {
        public:
                OpencvDNNPredictor(dlalg_jsons::ModelInfo &modelInfo);
                ~OpencvDNNPredictor() = default;       

                 // opencv模型处理
                int InitModel();
                //Deprecated 
                int PredictImp(const cv::Mat &img, std::vector<cv::Mat> &outputBlobs);
                int PredictImp(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outputBlobs);
                
             
        private:
             
                cv::dnn::Net net_;

        };

}

#endif