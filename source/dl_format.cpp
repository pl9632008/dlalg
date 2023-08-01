#include "dl_format.h"
#include "dl_common.h"


#include <glog/logging.h>
#include <fstream>


namespace yjh_deeplearning{



namespace dlalg_jsons {


void from_json(const json& j, PreprocessInfo& pi)
{

    
    if(j.find("img_height") != j.end())
    {
        j.at("img_height").get_to(pi.img_height);       
    }

    if(j.find("img_width") != j.end())
    {
        j.at("img_width").get_to(pi.img_width);       
    }
    
    if(j.find("img_channel") != j.end())
    {
        j.at("img_channel").get_to(pi.img_channel);       
    }
        
    if(j.find("img_mean") != j.end())
    {
        for(unsigned int i=0;i<j["img_mean"].size();i++)
        {           
            pi.img_mean.emplace_back(j.at("img_mean")[i]);
        }
    }

    if(j.find("short_size") != j.end())
    {
        j.at("short_size").get_to(pi.short_size);       
    }

    if(j.find("max_size") != j.end())
    {
        j.at("max_size").get_to(pi.max_size);       
    }

    if(j.find("padding") != j.end())
    {
        j.at("padding").get_to(pi.padding);
    }
   

    if(j.find("img_std") != j.end())
    {
        for(unsigned int i=0;i<j["img_std"].size();i++)
        {           
            pi.img_std.emplace_back(j.at("img_std")[i]);
        }
    }
      
    if(j.find("img_max_value") != j.end())
    {
        j.at("img_max_value").get_to(pi.img_max_value);       
    }
   
    if(j.find("is_rgb") != j.end())
    {
        j.at("is_rgb").get_to(pi.is_rgb);       
    }

    if(j.find("img_scale") != j.end())
    {
        j.at("img_scale").get_to(pi.img_scale);       
    }
}

void from_json(const json& j, ModelInfo& mi){
	j.at("model_name").get_to(mi.model_name);

    
         
    j.at("weight_path").get_to(mi.weight_path);

    if(j.find("auto_preprocess") != j.end())
    {
        for(unsigned int i=0;i<j["auto_preprocess"].size();i++)
        {
            mi.auto_preprocess.emplace_back(j.at("auto_preprocess")[i]);
        }	
    }
    
    if(j.find("preprocess_list") != j.end())
    {
        j.at("preprocess_list").get_to(mi.preprocess_list);
    }    
    
    if(j.find("onnxruntime_customop_library") != j.end())
    {
        j.at("onnxruntime_customop_library").get_to(mi.onnxruntime_customop_library);
    } 

    if(j.find("onnxruntime_intraOpNum") != j.end())
    {
        j.at("onnxruntime_intraOpNum").get_to(mi.onnxruntime_intraOpNum);
    }
    else
    {
        mi.onnxruntime_intraOpNum = 8;
    }
    

    if(j.find("batch_size") != j.end())
    {
        j.at("batch_size").get_to(mi.batch_size);
    }
    else
    {
        mi.batch_size = 1;
    } 

    

    if(j.find("model_type") != j.end())
    {
        j.at("model_type").get_to(mi.model_type);       
    }

    if(j.find("infer_engine") != j.end())
    {
        j.at("infer_engine").get_to(mi.infer_engine);
    }  

    if(j.find("cfg_path") != j.end())
    {
        j.at("cfg_path").get_to(mi.cfg_path);       
    }
   
    if(j.find("class_name") != j.end())
    {
        for(unsigned int i=0;i<j["class_name"].size();i++)
        {
            mi.class_name.emplace_back(j.at("class_name")[i]);
        }	
    }

    if(j.find("other_list") != j.end())
    {
        for(unsigned int i=0;i<j["other_list"].size();i++)
        {
            mi.other_list.emplace_back(j.at("other_list")[i]);
        }	
    }

     if(j.find("anchor") != j.end())
    {
        for(unsigned int i=0;i<j["anchor"].size();i++)
        {
            std::vector<float> tmp_anchor;
            for(unsigned int k=0;k<j["anchor"][i].size();k++)
            {
                tmp_anchor.emplace_back(j["anchor"][i][k]);
            }
            mi.anchor.emplace_back(tmp_anchor);
        }	
    }
    if(j.find("stride") != j.end())
    {
        for(unsigned int i=0;i<j["stride"].size();i++)
        {
            mi.stride.emplace_back(j["stride"][i]);
        }
    }
    
    if(j.find("class_thresh") != j.end())
    {
        mi.class_thresh = j.at("class_thresh").get<std::map<std::string, float>>();
    }

    if(j.find("other_conf_thresh") != j.end())
    {
        mi.other_conf_thresh = j.at("other_conf_thresh").get<std::map<std::string, float>>();
    }    

    if(j.find("other_conf") != j.end())
    {
        mi.other_conf = j.at("other_conf").get<std::map<std::string, std::string>>();
    }    
    
    
    if(j.find("name_input") != j.end())
    {
         for(unsigned int i=0;i<j["name_input"].size();i++)
        {
            mi.name_input.emplace_back(j.at("name_input")[i]);
        }	
    }

    if(j.find("name_output") != j.end())
    {
         for(unsigned int i=0;i<j["name_output"].size();i++)
        {
            mi.name_output.emplace_back(j.at("name_output")[i]);
        }	
    }
    
    if(j.find("conf_threshold") != j.end())
    {
        j.at("conf_threshold").get_to(mi.conf_threshold);       
    }

    if(j.find("iou_threshold") != j.end())
    {
        j.at("iou_threshold").get_to(mi.iou_threshold);       
    }

    if(j.find("obj_threshold") != j.end())
    {
        j.at("obj_threshold").get_to(mi.obj_threshold); 
    }    
    
    if(j.find("mask_threshold") != j.end())
    {
        j.at("mask_threshold").get_to(mi.mask_threshold);       
    }
  

    if(j.find("is_half") != j.end())
    {
        j.at("is_half").get_to(mi.is_half);       
    }
    else
    {
        mi.is_half = true;
    }

    if(j.find("is_dynamic_infer") != j.end())
    {
        j.at("is_dynamic_infer").get_to(mi.is_dynamic_infer);       
    }


    if(j.find("is_draw") != j.end())
    {
       j.at("is_draw").get_to(mi.is_draw);       
    } 

    if(j.find("multi_label") != j.end())
    {
       j.at("multi_label").get_to(mi.multi_label);       
    }   
    else
    {
         mi.multi_label = false;
    }
	
  }

}

 
int GetConfigFromJson(const std::string josn_file,dlalg_jsons::AlgInfo &algCfg)
{
	json j;
    std::ifstream jfile(josn_file,std::ios::in);
    if(!jfile.is_open())
    {   
        LOG(ERROR) << "can not open alg cfg file "<<josn_file;
        return YJH_AI_CFGFILE_NOFIND_ERROR;        
    }

	try{

		j=json::parse(jfile);
		j.at("alg_name").get_to(algCfg.alg_name);
        j.at("model_list").get_to(algCfg.model_list);		

	}catch (json::exception& e)
    {
        // output exception information
        LOG(ERROR) << "message: " << e.what() << " " << "exception id: " << e.id;
        return YJH_AI_CFGFILE_PARSE_ERROR;
    }catch (...) {
        LOG(ERROR) << "unknown json error or json file is not exist "<<josn_file;
        return YJH_AI_CFGFILE_PARSE_ERROR;
    }


	
	return DLSUCCESSED;


}


}
