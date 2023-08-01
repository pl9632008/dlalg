#include "dl_algorithm.h"
#include "dl_common.h"
#include "dl_algfactory.h"

#include <string>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>


namespace yjh_deeplearning {


class AIAlgorithm::AIAlgorithmImp {
	public:
		AIAlgorithmImp()=default;
		~AIAlgorithmImp();
		int Init(const std::string &model_path, int gpu_index);
		int Inference(const AIInputInfo &input_info,AIOutputInfo &output_info);
		int Inference(const std::vector<AIInputInfo> &input_list,std::vector<AIOutputInfo> &output_list);
      
	private:
		std::shared_ptr<BaseDLAlg> aialg_;
				
	};


AIAlgorithm::AIAlgorithmImp::~AIAlgorithmImp()
{
	if(aialg_ != nullptr)
	{
		aialg_->DeInit();
	}	
}

int AIAlgorithm::AIAlgorithmImp::Init(const std::string &alg_conf_file, int gpu_index)
{
	//初始化算法日志，执行一次
	
	if(!google::IsGoogleLoggingInitialized())
	{
		FLAGS_log_dir = "./";
		FLAGS_logbufsecs = 0;
		FLAGS_max_log_size = 2;
		google::InitGoogleLogging("YJHDLALG");
	}
		
	
	dlalg_jsons::AlgInfo algCfg;
	int ret;
	ret = GetConfigFromJson(alg_conf_file,algCfg);
	if(ret!=DLSUCCESSED)
	{
		LOG(ERROR) << "init json file failed";
		return ret;
	}
	aialg_ = DLAlgClassFactory::getInstance().getClassByName(algCfg.alg_name);
	if(aialg_ == nullptr)
	{
		LOG(ERROR) << "get alg instance failed";
		return YJH_AI_ALG_NOFIND_ERROR;
	}

	for(unsigned int i=0;i<algCfg.model_list.size();i++)
	{
		algCfg.model_list[i].gpu_index = gpu_index;
	}
	return aialg_->Init(algCfg);
}


int AIAlgorithm::AIAlgorithmImp::Inference(const AIInputInfo &input_info,AIOutputInfo &output_info)
{	
	if(aialg_ == nullptr)
	{
		return YJH_AI_ALG_INIT_ERROR;
	}
	return aialg_->ProcessPic(input_info,output_info);
}

int AIAlgorithm::AIAlgorithmImp::Inference(const std::vector<AIInputInfo> &input_list,std::vector<AIOutputInfo> &output_list)
{	
	if(aialg_ == nullptr)
	{
		return YJH_AI_ALG_INIT_ERROR;
	}
	return aialg_->ProcessPic(input_list,output_list);
}

int AIAlgorithm::Init(const std::string &alg_conf_file,int gpu_index)
{
	alg_imp = std::make_shared<AIAlgorithmImp>();
	return alg_imp->Init(alg_conf_file,gpu_index);	
}



int AIAlgorithm::Inference(const AIInputInfo &input_info,AIOutputInfo &output_info)
{
	return 	alg_imp->Inference(input_info,output_info);
}


int AIAlgorithm::Inference(const std::vector<AIInputInfo> &input_list,std::vector<AIOutputInfo> &output_list)
{
	return 	alg_imp->Inference(input_list,output_list);
}


std::shared_ptr<void> GetAlgorithmInstance()
{
    if(!google::IsGoogleLoggingInitialized())
	{
		FLAGS_log_dir = "./";
		FLAGS_logbufsecs = 0;
		FLAGS_max_log_size = 2;
		google::InitGoogleLogging("YJHDLALG");
	}
    //  LOG(ERROR) << "Get face recognition instance! ";
    return std::make_shared<yjh_deeplearning::AIAlgorithm>();

}


int InitAlgorithm(std::shared_ptr<void> instance,std::string alg_conf_file,int gpu_index)
{
     // LOG(ERROR) << "init face recognition start "<<alg_conf_file<<" "<<gpu_index;
    if(instance != nullptr)
    {
        std::shared_ptr<yjh_deeplearning::AIAlgorithm> yjh_alg = std::static_pointer_cast<yjh_deeplearning::AIAlgorithm>(instance);
        return yjh_alg->Init(alg_conf_file,gpu_index);
    }
    else
    {
        return yjh_deeplearning::DLFAILED;
    }
}

int AlgorithmInference(std::shared_ptr<void> instance,AIInputInfo &input_info,AIOutputInfo &output_info)
{
    if(instance != nullptr )
    {
        std::shared_ptr<yjh_deeplearning::AIAlgorithm> yjh_alg = std::static_pointer_cast<yjh_deeplearning::AIAlgorithm>(instance);               
               
        return yjh_alg->Inference(input_info,output_info);   
    }
    else
    {
        LOG(ERROR) << "instance is nullptr or output is nullptr";
        return yjh_deeplearning::DLFAILED;
    }
    return DLSUCCESSED;
}

int AlgorithmBatchInference(void *instance,std::vector<AIInputInfo> &inputs_info,std::vector<AIOutputInfo> &outputs_info)
{
    if(instance != nullptr )
    {
         yjh_deeplearning::AIAlgorithm* yjh_alg = reinterpret_cast<yjh_deeplearning::AIAlgorithm*>(instance);
        
        return yjh_alg->Inference(inputs_info,outputs_info);        
      
    }
    else
    {
        LOG(ERROR) << "instance is nullptr or output is nullptr";
        return yjh_deeplearning::DLFAILED;
    }
    return DLSUCCESSED;
       
}


}