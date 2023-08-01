#include "dl_normalize.h"
#include "dl_common.h"
#include <glog/logging.h>


namespace yjh_deeplearning{

int Normalize::Apply(std::vector<cv::Mat> &imgs,std::vector<dlalg_jsons::PreprocessInfo> &preprocess_list)
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
                imgs[i].convertTo(image, CV_32F, 1.0 / preprocess_list[index].img_max_value);
                if(preprocess_list[index].img_mean.size() !=0 )
                {
                    cv::subtract(image, cv::Scalar(preprocess_list[index].img_mean[0], preprocess_list[index].img_mean[1], preprocess_list[index].img_mean[2]), image);
                }
			    if(preprocess_list[index].img_std.size() !=0 )
                {
                    cv::divide(image, cv::Scalar(preprocess_list[index].img_std[0], preprocess_list[index].img_std[1], preprocess_list[index].img_std[2]), image);  
                }
			    
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
        LOG(ERROR)<<"normalize parameter error";
        return DLFAILED;
    }
    
}


REGISTERTRANSFORM(normalize, Normalize);

}