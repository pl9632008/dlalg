#include "dl_pad.h"
#include "dl_common.h"
#include <glog/logging.h>


namespace yjh_deeplearning{

int Pad::Apply(std::vector<cv::Mat> &imgs,std::vector<dlalg_jsons::PreprocessInfo> &preprocess_list)
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
                int w, h, x, y;
                float r_w = preprocess_list[index].img_width / (imgs[i].cols*1.0);
                float r_h = preprocess_list[index].img_height / (imgs[i].rows*1.0);
                if (r_h > r_w) {
                    w = preprocess_list[index].img_width;
                    h = r_w * imgs[i].rows;
                    x = 0;
                    y = (preprocess_list[index].img_height - h) / 2;
                } else {
                    w = r_h * imgs[i].cols;
                    h = preprocess_list[index].img_height;
                    x = (preprocess_list[index].img_width - w) / 2;
                    y = 0;
                }
                cv::Mat re(h, w, CV_8UC3);
                cv::resize(imgs[i], re, re.size(), 0, 0, cv::INTER_LINEAR);
                cv::Mat out(preprocess_list[index].img_height, preprocess_list[index].img_width, CV_8UC3, cv::Scalar(128, 128, 128));
                re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));        
                image = out;
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
        LOG(ERROR)<<"pad parameter error";
        return YJH_AI_INPUT_IMGPREPROCESS_ERROR;
    }
    
}

REGISTERTRANSFORM(pad, Pad);

}