#include "meter_recognition.h"
#include "meter_format.h"
#include "dl_algorithm.h"
#include "dl_common.h"
#include <string>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>


std::map<std::string,Device_Type> DefectMapList={ {"xy",Smoking},{"wcgz",NoOveralls},{"wcaqm",NoHelmet},{"wggm",UnclosedDoor},{"bpps",DialsDamaged},{"sly_dmyw",OilLeakage} };

std::string GetMeterSDKVersion()
{      
    std::string SDKVersionInfo = "AI SDK Version: 0.0.1\n";                                 
    return SDKVersionInfo;
}


int SetWorkMode(void *engine, int mode_type)
{
    return 0;
}

int MeterLoadModel(const std::string &model_path, int gpu_index, void *&engine)
{
     if(!google::IsGoogleLoggingInitialized())
	{
		FLAGS_log_dir = "./";
		FLAGS_logbufsecs = 0;
		FLAGS_max_log_size = 2;
		google::InitGoogleLogging("YJHDLALG");
	}
    //  LOG(ERROR) << "Get face recognition instance! ";
    yjh_deeplearning::AIAlgorithm  *yjh_alg = new yjh_deeplearning::AIAlgorithm();
    // LOG(ERROR) << "init face recognition start "<<alg_conf_file<<" "<<gpu_index;
    if(yjh_alg != nullptr && yjh_alg->Init(model_path,gpu_index) == yjh_deeplearning::DLSUCCESSED)
    {   
        engine = reinterpret_cast<void*>(yjh_alg);
        return 0;
    }
    else
    {
        return -1;
    }
}



int MeterRecognition(void *engine, void *image_data, void *depth_data, const std::string &config_path, std::vector<DETECT_RESULT> &detect_result)
{
   if(engine != nullptr && image_data != nullptr)
    {
        int ret;
        Meter_jsons::MeterConfigInfo meterCfg;
        ret = GetMeterConfigFromJson(config_path,meterCfg);
        if(ret!=yjh_deeplearning::DLSUCCESSED)
        {
            LOG(ERROR) << "read meterCfg json file failed";
            return ret;
        }

        yjh_deeplearning::AIAlgorithm* yjh_alg = reinterpret_cast<yjh_deeplearning::AIAlgorithm*>(engine);
        
        yjh_deeplearning::AIInputInfo input_info;
        yjh_deeplearning::AIOutputInfo output_info;
        
        input_info.src_mat.emplace_back(std::make_shared<cv::Mat>(*reinterpret_cast<cv::Mat*>(image_data)));  
        ret = yjh_alg->Inference(input_info,output_info); 
        if(ret != yjh_deeplearning::DLSUCCESSED)
        {
            return ret;
        }
        detect_result.clear();
        DETECT_RESULT tmp;
        AlgPoint tmp_point;      
        for(unsigned int i=0;i<output_info.result_list.size();i++)
        {           
            if(std::find(std::begin(meterCfg.recognition_list), std::end(meterCfg.recognition_list), output_info.result_list[i].value) == std::end(meterCfg.recognition_list))               
            {
                continue;
            }
            tmp.id = DefectMapList[output_info.result_list[i].value];
            tmp.points.clear();
            tmp_point.x = output_info.result_list[i].center_x - output_info.result_list[i].width/2;
            tmp_point.y = output_info.result_list[i].center_y - output_info.result_list[i].height/2;
            tmp.points.emplace_back(tmp_point);
            tmp_point.x = output_info.result_list[i].center_x + output_info.result_list[i].width/2;
            tmp_point.y = output_info.result_list[i].center_y - output_info.result_list[i].height/2;
            tmp.points.emplace_back(tmp_point);
            tmp_point.x = output_info.result_list[i].center_x + output_info.result_list[i].width/2;
            tmp_point.y = output_info.result_list[i].center_y + output_info.result_list[i].height/2;
            tmp.points.emplace_back(tmp_point);
            tmp_point.x = output_info.result_list[i].center_x - output_info.result_list[i].width/2;
            tmp_point.y = output_info.result_list[i].center_y + output_info.result_list[i].height/2;
            tmp.points.emplace_back(tmp_point);
            tmp.score = output_info.result_list[i].score;
            tmp.value = "bad";
            detect_result.emplace_back(tmp);
        }

        if(detect_result.size() == 0)
        {
            for(unsigned int i=0;i<meterCfg.recognition_list.size();i++)
            {
                tmp.id = DefectMapList[meterCfg.recognition_list[i]];
                tmp.points.clear();
                tmp.value = "good";
                detect_result.emplace_back(tmp);
            }
        }
        return yjh_deeplearning::DLSUCCESSED;

    }
    else
    {
        return yjh_deeplearning::DLFAILED;
    } 

}


int MeterReleaseModel(void *&engine)
{
    if(engine != nullptr)
    {        
        delete reinterpret_cast<yjh_deeplearning::AIAlgorithm*>(engine);;
        engine = nullptr;
    }
    return 0;
}