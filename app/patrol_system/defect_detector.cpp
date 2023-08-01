#include "defect_detector.h"
#include "dl_algorithm.h"
#include "dl_common.h"
#include "nlohmann/json.hpp"
#include <string>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace yjh_deeplearning{

class PatrolSystemDefectRecognition::PatrolSystemDefectRecognitionImp
{
    public:
        PatrolSystemDefectRecognitionImp()=default;
        ~PatrolSystemDefectRecognitionImp()=default;
        int Init(const std::string &alg_conf_file,int gpu_index);
        int ProcessImage(const std::vector<std::string> &imgs_path, std::vector<AIOutputInfo> &detect_result, std::vector<std::string> &img_info);
        int ProcessImageJson(const std::vector<std::string> &imgs_path, std::string &detect_json);
    private:
        AIAlgorithm dlalg_;
        int ToJson(const std::vector<std::string> &imgs_path, const std::vector<AIOutputInfo> &detect_results, nlohmann::json &jsonOutputs);
        std::vector<std::string> img_flag_;
};
    

int PatrolSystemDefectRecognition::PatrolSystemDefectRecognitionImp::Init(const std::string &alg_conf_file, int gpu_index)
{
    return dlalg_.Init(alg_conf_file, gpu_index);
}

int PatrolSystemDefectRecognition::PatrolSystemDefectRecognitionImp::ProcessImage(const std::vector<std::string> &imgs_path, std::vector<AIOutputInfo> &detect_result, std::vector<std::string> &img_info)
{
    if(imgs_path.size() <= 0)
    {
        LOG(ERROR) << "img input path is empty";
        return YJH_AI_IMAGE_INFO_ERROR;  // 返回 -5
    }
    yjh_deeplearning::AIInputInfo input_info;
    yjh_deeplearning::AIOutputInfo output_info;
    std::vector<yjh_deeplearning::AIInputInfo> input_list;

    // 读取图片列表，将图片数据放入input_list中
    img_flag_.clear();
    input_list.clear();
    for(unsigned int i=0; i<imgs_path.size(); i++)
    {
        LOG(INFO) << "image path: "<< imgs_path[i];
        cv::Mat cv_img = cv::imread(imgs_path[i]);
        if(cv_img.data == nullptr)
        {
            LOG(ERROR) << "read image error, image path: " << imgs_path[i];
            img_flag_.emplace_back("ImgError");
            continue;
        }
        input_info.src_mat.clear();
        input_info.src_mat.emplace_back(std::make_shared<cv::Mat>(cv_img));
        input_list.emplace_back(input_info);
        img_flag_.emplace_back("OK");
    }
    img_info = img_flag_;

    int ret;
    if(input_list.size() != 0)
    {
        // 将图片数据送入模型中推理
        detect_result.clear();
        if(input_list.size() == 1)
        {
            LOG(INFO) << "single img inference ...";
            ret = dlalg_.Inference(input_list[0], output_info);
            detect_result.emplace_back(output_info);
            LOG(INFO) << "single img inference finished ...";
        }
        else
        {
            LOG(INFO) << "batch imgs inference ...";
            ret = dlalg_.Inference(input_list, detect_result);
            LOG(INFO) << "batch imgs inference finished ...";
        }

        if(ret != 0 )
        {
            return ret;
        }   
    }

    return DLSUCCESSED;
}

int PatrolSystemDefectRecognition::PatrolSystemDefectRecognitionImp::ToJson(const std::vector<std::string> &imgs_path, const std::vector<AIOutputInfo> &detect_results, nlohmann::json &jsonOutputs)
{
    if(img_flag_.size() != imgs_path.size())
    {
        LOG(ERROR) <<  "img flag num: " << img_flag_.size() << "!= input images num: " << imgs_path.size();
        return YJH_AI_INPUT_ERROR;  // 返回 -4
    }
    
    int err = 0;
    for(unsigned int i = 0; i < imgs_path.size(); i++)
    {
        nlohmann::json jsonDefect; 
        if(img_flag_[i] == "ImgError")
        {
            jsonDefect["code"] = img_flag_[i];
            jsonDefect["objects"] = {""};
            // LOG(INFO) << "jsonDefect = "  << jsonDefect.dump() << std::endl;
            err = err + 1;
        }
        else
        {
            int cor = i - err;
            nlohmann::json jsonObj;
            for(unsigned int j = 0; j < detect_results[cor].result_list.size(); j++)
            {
                nlohmann::json jsonTmp;
                jsonTmp["class"] = detect_results[cor].result_list[j].value;
                jsonTmp["score"] = static_cast<float>(detect_results[cor].result_list[j].score);
                jsonTmp["box"] = {detect_results[cor].result_list[j].center_x,
                                  detect_results[cor].result_list[j].center_y,
                                  detect_results[cor].result_list[j].width,
                                  detect_results[cor].result_list[j].height};
                
                jsonObj.emplace_back(jsonTmp);
            }
            jsonDefect["objects"] = jsonObj;
            jsonDefect["code"] = img_flag_[i];
            // LOG(INFO) << "jsonDefect = "  << jsonDefect.dump() << std::endl;

        }
        jsonOutputs[imgs_path[i]] = jsonDefect; 
        // LOG(INFO) << "jsonOutputs = "  << jsonOutputs.dump() << std::endl;    
    }
    return yjh_deeplearning::DLSUCCESSED;
}

int PatrolSystemDefectRecognition::PatrolSystemDefectRecognitionImp::ProcessImageJson(const std::vector<std::string> &imgs_path, std::string &detect_json)
{
    if(imgs_path.size() <= 0)
    {
        LOG(ERROR) << "img input path is empty";
        return YJH_AI_IMAGE_INFO_ERROR;  // 返回 -5
    }
    yjh_deeplearning::AIInputInfo input_info;
    yjh_deeplearning::AIOutputInfo output_info;
    std::vector<yjh_deeplearning::AIInputInfo> input_list;
    std::vector<yjh_deeplearning::AIOutputInfo> detect_result;

    // 读取图片列表，将图片数据放入input_list中
    img_flag_.clear();
    input_list.clear();
    for(unsigned int i=0; i<imgs_path.size(); i++)
    {
        LOG(INFO) << "image path: "<< imgs_path[i];
        cv::Mat cv_img = cv::imread(imgs_path[i]);

        if(cv_img.data == nullptr)
        {
            LOG(ERROR) << "read image  error, image path: " << imgs_path[i];
            img_flag_.emplace_back("ImgError");
            continue;
        }

        input_info.src_mat.clear();
        input_info.src_mat.emplace_back(std::make_shared<cv::Mat>(cv_img));
        input_list.emplace_back(input_info);
        img_flag_.emplace_back("OK");
    }

    int ret;
    if(input_list.size() != 0)
    {
        // 将图片数据送入模型中推理
        detect_result.clear();

        if(input_list.size() == 1)
        {
            LOG(INFO) << "single img inference ...";
            ret = dlalg_.Inference(input_list[0], output_info);
            detect_result.emplace_back(output_info);
            LOG(INFO) << "single img inference finished ...";
        }
        else
        {
            LOG(INFO) << "batch imgs inference ...";
            ret = dlalg_.Inference(input_list, detect_result);
            LOG(INFO) << "batch imgs inference finished ...";
        }

        if(ret != 0 )
        {
            return ret;
        }   
    }

    nlohmann::json json_results;
    ret = ToJson(imgs_path, detect_result, json_results);
    if(ret != 0 )
    {
        return ret;
    }  

    detect_json = json_results.dump(4);
    // LOG(INFO) << "detect_json = "  << detect_json << std::endl;  

    return DLSUCCESSED;
}


int PatrolSystemDefectRecognition::Init(std::string alg_conf_file,int gpu_index)
{
    defect_alg_imp_ = std::make_shared<PatrolSystemDefectRecognitionImp>();
    return defect_alg_imp_->Init(alg_conf_file, gpu_index);
}

int PatrolSystemDefectRecognition::ProcessImage(const std::vector<std::string> &imgs_path, std::vector<AIOutputInfo> &detect_result, std::vector<std::string> &img_info)
{
    return defect_alg_imp_->ProcessImage(imgs_path, detect_result, img_info);
}

int PatrolSystemDefectRecognition::ProcessImageJson(const std::vector<std::string> &imgs_path, std::string &detect_json)
{
    return defect_alg_imp_->ProcessImageJson(imgs_path, detect_json);
}


} // namespace yjh_deeplearning






