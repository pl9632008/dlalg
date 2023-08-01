#include "dl_scrfd_model.h"
#include "dl_common.h"
#include "dl_opencvdnn.h"
#include "dl_onnxruntime.h"
#include <glog/logging.h>


namespace yjh_deeplearning{

SCRFDModel::SCRFDModel(dlalg_jsons::ModelInfo &modelInfo)
{
    preprocess_list_ = modelInfo.preprocess_list;
    stride_ = modelInfo.stride;
    infer_engine_ = modelInfo.infer_engine;
    if(infer_engine_ == INFER_ENGINE_ORT)
    {
        model_ = std::make_shared<ONNXRuntimePredictor>(modelInfo); 
    }
    else
    {
        model_ = std::make_shared<OpencvDNNPredictor>(modelInfo); 
    }
         
    conf_threshold_ = modelInfo.conf_threshold;
    iou_threshold_ = modelInfo.iou_threshold; 
      
 }

int SCRFDModel::InitModel()
{

    if(preprocess_list_.size() == 0 || stride_.size() != 3 || model_ == nullptr)
    {
        LOG(ERROR) << "preprocess config error or model is nullptr";
        return YJH_AI_PARAMETER_ERROR;
    }
    img_height_ = preprocess_list_[0].img_height;
    img_width_ = preprocess_list_[0].img_width;
    return model_->InitModel();
}


int SCRFDModel::ModelPredict(const cv::Mat &img, AIOutputInfo &output_info)
{      
    cv::Mat image = img.clone();  
    CHECK_SUCCESS(PreProcess(image));
    CHECK_SUCCESS(model_->Predict(image,out_tensors_));
    if(infer_engine_ == INFER_ENGINE_ORT)
    {        
        std::vector<cv::Mat> out_tensors = out_tensors_;
        for(int i=0;i<3;i++)
        {
            for(int j=0;j<3;j++)
            {
                out_tensors_[j+i*3] = out_tensors[j*3+i];  
            }                     
        }
    }   
    CHECK_SUCCESS(PostProcess(out_tensors_,output_info));
    
    return DLSUCCESSED;
    
}

int SCRFDModel::PreProcess(cv::Mat &img)
{   
    int srch = img.rows, srcw = img.cols;
	newh_ = img_height_;
	neww_ = img_width_;
    padw_=padh_=0;
	try{
        if (srch != srcw)
        {
            float hw_scale = (float)srch / srcw;
            if (hw_scale > 1)
            {
                newh_ = img_height_;
                neww_ = int(img_width_ / hw_scale);
                resize(img, img, cv::Size(neww_, newh_), cv::INTER_AREA);
                padw_ = int((img_width_ - neww_) * 0.5);
                cv::copyMakeBorder(img, img, 0, 0, padw_, img_width_ - neww_ - padw_, cv::BORDER_CONSTANT, 0);
            }
            else
            {
                newh_ = (int)img_height_ * hw_scale;
                neww_ = img_width_;
                resize(img, img, cv::Size(neww_, newh_), cv::INTER_AREA);
                padh_ = (int)(img_height_ - newh_) * 0.5;
                cv::copyMakeBorder(img, img, padh_, img_height_ - newh_ - padh_, 0, 0, cv::BORDER_CONSTANT, 0);
            }
        }
        else
        {
            resize(img, img, cv::Size(neww_, newh_), cv::INTER_AREA);
        }
        ratioh_ = (float)srch / newh_;
        ratiow_ = (float)srcw / neww_;
        // cv::imwrite("preim.jpg",img);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        CHECK_SUCCESS(model_->GeneralPreProcess(img));
    }
    catch(cv::Exception& e)
    {
        // output exception information
        LOG(ERROR) << "message: " << e.what() ;
        return YJH_AI_INPUT_IMGPREPROCESS_ERROR;
    }

	return DLSUCCESSED;
}
int SCRFDModel::PostProcess(std::vector<cv::Mat> &outs,AIOutputInfo &output_info)
{    
    std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<std::vector<int>> landmarks;
	int n = 0, i = 0, j = 0, k = 0, l = 0;
	for (n = 0; n < 3; n++)   
	{
		int num_grid_x = (int)(img_width_ / stride_[n]);
		int num_grid_y = (int)(img_height_ / stride_[n]);
		float* pdata_score = (float*)outs[n * 3].data;  ///score
		float* pdata_bbox = (float*)outs[n * 3 + 1].data;  ///bounding box
		float* pdata_kps = (float*)outs[n * 3 + 2].data;  ///face landmark
		for (i = 0; i < num_grid_y; i++)
		{
			for (j = 0; j < num_grid_x; j++)
			{
				for (k = 0; k < 2; k++)
				{
					if (pdata_score[0] > conf_threshold_)
					{			
                        // LOG(ERROR)<<"pdata_bbox "<<pdata_bbox[0]<<" "<<pdata_bbox[1]<<" "<<pdata_bbox[2]<<" "<<pdata_bbox[3]<<" "<<stride_[n]<<" "<<n<<" "<<ratiow_<<" "<<ratioh_<<" "<<padw_<<" "<<padh_;			
						const int xmin = (int)(((j - pdata_bbox[0]) * stride_[n] - padw_) * ratiow_);
						const int ymin = (int)(((i - pdata_bbox[1]) * stride_[n] - padh_) * ratioh_);
						const int width = (int)((pdata_bbox[2] + pdata_bbox[0])*stride_[n] * ratiow_);
						const int height = (int)((pdata_bbox[3] + pdata_bbox[1])*stride_[n] * ratioh_);
						confidences.push_back(pdata_score[0]);
						boxes.push_back(cv::Rect(xmin, ymin, width, height));                       
						std::vector<int> landmark(10, 0);
						for (l = 0; l < 10; l+=2)
						{
							landmark[l] = (int)(((j + pdata_kps[l]) * stride_[n] - padw_) * ratiow_);
							landmark[l + 1] = (int)(((i + pdata_kps[l + 1]) * stride_[n] - padh_) * ratioh_);
						}
						landmarks.push_back(landmark);
					}
					pdata_score++;
					pdata_bbox += 4;
					pdata_kps += 10;
				}
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, iou_threshold_, indices);
    AIResult tmpreslut;
    output_info.result_list.clear();
	for (i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];		
        // LOG(ERROR)<<"Rect "<<boxes[idx].x<<" "<<boxes[idx].y<<" "<<boxes[idx].width<<"  "<<boxes[idx].height;
        tmpreslut.center_x = boxes[idx].x + boxes[idx].width/2;
        tmpreslut.center_y = boxes[idx].y +  boxes[idx].height/2;
        tmpreslut.width = boxes[idx].width;
        tmpreslut.height = boxes[idx].height;      
		tmpreslut.key_points.clear();
		for (k = 0; k < 10; k+=2)
		{
            tmpreslut.key_points.emplace_back(std::make_pair(landmarks[idx][k], landmarks[idx][k + 1]));			
		}
        output_info.result_list.emplace_back(tmpreslut);		
	}
    return DLSUCCESSED;
}

void SCRFDModel::DeInitModel()
{       
    if(model_ != nullptr)
    {
        model_->DeInitModel();
    }
       
}

}