#include "dl_bgr2rgb.h"
#include "dl_common.h"
#include <glog/logging.h>


namespace yjh_deeplearning{

int BGR2RGB::Apply(std::vector<cv::Mat> &imgs,std::vector<dlalg_jsons::PreprocessInfo> &preprocess_list)
{
    if(imgs.size() == preprocess_list.size() ||  preprocess_list.size() == 1)
    {  
        for(unsigned int i=0;i<imgs.size();i++)
        {   
            cv::Mat image;            
            try
            {        
                cv::cvtColor(imgs[i], image, cv::COLOR_BGR2RGB);               
            }
            catch (cv::Exception &e)
            {
                // output exception information
                LOG(ERROR) << "message: " << e.what();
                return YJH_AI_INPUT_IMGPREPROCESS_ERROR;
            }  
            imgs[i] = image;
        }
        return DLSUCCESSED;
    }
    else
    {
        LOG(ERROR)<<"resize parameter error";
        return YJH_AI_INPUT_IMGPREPROCESS_ERROR;
    }
    
}


REGISTERTRANSFORM(bgr2rgb, BGR2RGB);

}