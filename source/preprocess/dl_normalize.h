#ifndef DL_NORMALIZE_H
#define DL_NORMALIZE_H

#include "dl_transform.h"



namespace yjh_deeplearning{

class Normalize:public Transform{
    public:
        int Apply(std::vector<cv::Mat> &imgs,std::vector<dlalg_jsons::PreprocessInfo> &preprocess_list);
};




}

#endif