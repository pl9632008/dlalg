#include "dl_resize.h"
#include "dl_common.h"
#include <glog/logging.h>


namespace yjh_deeplearning{

int Resize::Apply(std::vector<cv::Mat> &imgs,std::vector<dlalg_jsons::PreprocessInfo> &preprocess_list)
{
    if(imgs.size() == preprocess_list.size() ||  preprocess_list.size() == 1)
    {
        unsigned int index=0;
        for(unsigned int i=0;i<imgs.size();i++)
        {   
            cv::Mat image;
            if(imgs.size() == preprocess_list.size())
            {
                index = i;       
            }
            try
            {                   
                cv::resize(imgs[i], image, cv::Size(preprocess_list[index].img_width, preprocess_list[index].img_height), 0, 0, cv::INTER_LINEAR);
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


REGISTERTRANSFORM(resize, Resize);

}