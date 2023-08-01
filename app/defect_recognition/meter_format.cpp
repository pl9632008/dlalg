#include "meter_format.h"
#include "dl_common.h"


#include <glog/logging.h>
#include <fstream>




 
int GetMeterConfigFromJson(const std::string josn_file,Meter_jsons::MeterConfigInfo &meterCfg)
{
	json j;
    std::ifstream jfile(josn_file,std::ios::in);
    if(!jfile.is_open())
    {   
        LOG(ERROR) << "can not open meter cfg file "<<josn_file;
        return yjh_deeplearning::YJH_AI_CFGFILE_NOFIND_ERROR;        
    }

	try{
        meterCfg.recognition_list.clear();
		j=json::parse(jfile);	
        if(j.find("recognition_list") != j.end())
        {
            for(unsigned int i=0;i<j["recognition_list"].size();i++)
            {
                meterCfg.recognition_list.emplace_back(j.at("recognition_list")[i]);
            }	
        }

	}catch (json::exception& e)
    {
        // output exception information
        LOG(ERROR) << "message: " << e.what() << " " << "exception id: " << e.id;
        return yjh_deeplearning::YJH_AI_CFGFILE_PARSE_ERROR;
    }catch (...) {
        LOG(ERROR) << "unknown json error or json file is not exist";
        return yjh_deeplearning::YJH_AI_CFGFILE_PARSE_ERROR;
    }


	
	return yjh_deeplearning::DLSUCCESSED;


}



