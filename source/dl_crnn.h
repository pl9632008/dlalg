#ifndef DL_CRNN_H
#define DL_CRNN_H

#include "dl_crnn_model.h"
#include "dl_format.h"
#include "dl_algfactory.h"

#include <opencv2/opencv.hpp>


namespace yjh_deeplearning{





class CRNNtAlg:public BaseDLAlg{
    public:
        
        int Init(dlalg_jsons::AlgInfo &algInfo);
		int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info);        
        int DeInit();
        // int LibTorchInit();
        // int LibTorchDetect(); 
    
    private:

        std::shared_ptr<CRNNModel> crnn_model_;    
       
   
 
 };
     
  
    
}



#endif