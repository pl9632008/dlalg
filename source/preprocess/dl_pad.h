#ifndef DL_PAD_H
#define DL_PAD_H

#include "dl_transform.h"



namespace yjh_deeplearning{

class Pad:public Transform{
    public:
        int Apply(std::vector<cv::Mat> &imgs,std::vector<dlalg_jsons::PreprocessInfo> &preprocess_list);
};




}

#endif