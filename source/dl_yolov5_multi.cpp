#include "dl_yolov5_multi.h"
#include "dl_common.h"

#include <glog/logging.h>

namespace yjh_deeplearning{



int Yolov5AlgMulti::Init(dlalg_jsons::AlgInfo &algInfo)
{
       
    if(algInfo.model_list.size() == 0 || algInfo.model_list[0].class_name.size() <= 0)
    {
        LOG(ERROR) << "yolo v5 can not set class" ;
        return YJH_AI_ALG_INIT_ERROR;  
    }

    // 按顺序获取配置文件中模型相应配置参数
    std::vector<std::string> tmp_class_name;
    std::map<std::string,float> tmp_class_thresh;
    std::shared_ptr<Yolov5Model> tmp_yolov5_model;
    for(unsigned int i=0;i<algInfo.model_list.size();i++)
    {
        tmp_class_name = algInfo.model_list[i].class_name;
        multi_class_name_.emplace_back(tmp_class_name);
        tmp_class_thresh = (algInfo.model_list[i].class_thresh);
        multi_class_thresh_.emplace_back(tmp_class_thresh);
        tmp_yolov5_model = std::make_shared<Yolov5Model>(algInfo.model_list[i]);
        CHECK_SUCCESS(tmp_yolov5_model->InitModel());
        multi_yolov5_model_.emplace_back(tmp_yolov5_model);
    }
    
    return DLSUCCESSED;
 }

int Yolov5AlgMulti::ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)
{
    if(input_info.src_mat.size() ==0 || input_info.src_mat[0] == nullptr)
    {
        return YJH_AI_INPUT_IMG_ERROR;   // 返回 -50
    }

    multi_detection_results_.clear();
    std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_info.src_mat[0]);    
    std::vector<DetectionObj> tmp_detection_results;
    // 图片送入模型依次推理，结果按顺序存储在列表multi_detection_results_中，即模型multi_yolov5_model_[n]对应输出结果是multi_detection_results_[n]
    for(unsigned int i=0;i<multi_yolov5_model_.size();i++)
    {
        CHECK_SUCCESS(multi_yolov5_model_[i]->ModelPredict(*ori_img, tmp_detection_results));
        multi_detection_results_.emplace_back(tmp_detection_results);
    }
    
    // 单张图片多个模型推理结果整合输出结果
    output_info.result_list.clear();
    AIResult result;
    for(unsigned i=0;i<multi_detection_results_.size();i++)
    {   
        for(unsigned int j=0;j<multi_detection_results_[i].size();j++)
        {
            if(multi_detection_results_[i][j].class_idx < multi_class_name_[i].size())
            {
                result.value = multi_class_name_[i][multi_detection_results_[i][j].class_idx];
                result.score = multi_detection_results_[i][j].score;
                if(multi_class_thresh_[i].find(result.value) != multi_class_thresh_[i].end() && result.score < multi_class_thresh_[i].find(result.value)->second)
                {
                    continue;
                }
                result.center_x = multi_detection_results_[i][j].center_x;
                result.center_y = multi_detection_results_[i][j].center_y;
                result.width = multi_detection_results_[i][j].width;
                result.height = multi_detection_results_[i][j].height;
                output_info.result_list.emplace_back(result);
            }
            else
            {
                LOG(ERROR) << "class idx greater than class name size";
            }
        }
    }
   
    return DLSUCCESSED;
}

int Yolov5AlgMulti::ProcessPic(const std::vector<AIInputInfo> &input_list,std::vector<AIOutputInfo> &output_list)
{   
    std::vector<cv::Mat> imgs;
    for(unsigned int i=0;i<input_list.size();i++)
    {
        if(input_list[i].src_mat.size() ==0 || input_list[i].src_mat[0] == nullptr)
        {
            return YJH_AI_INPUT_IMG_ERROR;  // 返回 -50
        }
        std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_list[i].src_mat[0]);       
        imgs.emplace_back(*ori_img);
    }  

    total_detection_results_.clear();
    std::vector<std::vector<DetectionObj>> tmp_total_detection_results;
    for(unsigned int i=0;i<multi_yolov5_model_.size();i++)
    {
        CHECK_SUCCESS(multi_yolov5_model_[i]->ModelPredict(imgs,tmp_total_detection_results));
        total_detection_results_.emplace_back(tmp_total_detection_results);
    }

    // 多张图片，多个模型推理结果整合输出结果
    output_list.clear();
    // LOG(ERROR) << "message: "<<total_detection_results_.size() ;
    AIResult result;
    AIOutputInfo result_output;
    // i代表模型的索引，j代表图片的索引，k代表图片推理结果的索引
    for(unsigned i = 0;i<total_detection_results_.size();i++)
    {   
        // 模型i多张图片推理结果
        for(unsigned int j=0;j<input_list.size();j++)
        {
            result_output.result_list.clear();
            for(unsigned int k=0;k<total_detection_results_[i][j].size();k++)
            {
                // 图片j推理结果
                if(total_detection_results_[i][j][k].class_idx<multi_class_name_[i].size())
                {
                    // 推理结果
                    result.value = multi_class_name_[i][total_detection_results_[i][j][k].class_idx];
                    result.score = total_detection_results_[i][j][k].score;
                    if(multi_class_thresh_[i].find(result.value) != multi_class_thresh_[i].end() && result.score < multi_class_thresh_[i].find(result.value)->second)
                    {
                        continue;
                    }
                    result.center_x = total_detection_results_[i][j][k].center_x;
                    result.center_y = total_detection_results_[i][j][k].center_y;
                    result.width = total_detection_results_[i][j][k].width;
                    result.height = total_detection_results_[i][j][k].height;
                    result_output.result_list.emplace_back(result);
                }
                else
                {
                    LOG(ERROR) << "class idx greater than class name size";
                }
            }

            if(i == 0)
            {
                output_list.emplace_back(result_output);
            }
            else
            {
                for(unsigned int l=0;l<result_output.result_list.size();l++)
                {
                    output_list[j].result_list.emplace_back(result_output.result_list[l]);
                }
            }
        }       
        
    }

    return DLSUCCESSED;
}

int Yolov5AlgMulti::DeInit()
{
    for(unsigned int i=0;i<multi_yolov5_model_.size();i++)
        multi_yolov5_model_[i]->DeInitModel();
    return DLSUCCESSED;
}


REGISTERALG(yolov5_dlalg_multi, Yolov5AlgMulti);


}