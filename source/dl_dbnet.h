#ifndef DL_DBNET_H
#define DL_DBNET_H

#include "dl_dbnet_model.h"
#include "dl_format.h"
#include "dl_algfactory.h"

#include <opencv2/opencv.hpp>


namespace yjh_deeplearning{





class DbnetAlg:public BaseDLAlg{
    public:
        
        int Init(dlalg_jsons::AlgInfo &algInfo);
		int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info);        
        int DeInit();
        // int LibTorchInit();
        // int LibTorchDetect(); 
    
    private:

        std::shared_ptr<DbnetModel> dbnet_model_;    
        std::vector<TextBox> detection_boxes_;
   
 
 };
     
  
    
}



#endif