#ifndef DL_BGR2RGB_H
#define DL_BGR2RGB_H

#include "dl_transform.h"



namespace yjh_deeplearning{

class BGR2RGB:public Transform{
    public:
        int Apply(std::vector<cv::Mat> &imgs,std::vector<dlalg_jsons::PreprocessInfo> &preprocess_list);
};


}

#endif