#ifndef DL_RESIZE_H
#define DL_RESIZE_H

#include "dl_transform.h"



namespace yjh_deeplearning{

class Resize:public Transform{
    public:
        int Apply(std::vector<cv::Mat> &imgs,std::vector<dlalg_jsons::PreprocessInfo> &preprocess_list);
};




}

#endif