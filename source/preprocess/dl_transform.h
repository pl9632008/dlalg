#ifndef DL_TRANSFORM_H
#define DL_TRANSFORM_H

#include "dl_format.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <map>
#include <mutex>
#include <functional>


namespace yjh_deeplearning{

class  Transform {
 public:
  virtual ~Transform() = default;
  virtual int Apply(std::vector<cv::Mat> &imgs,std::vector<dlalg_jsons::PreprocessInfo> &preprocess_list) = 0;
};

using CreateTransform_Func = std::function<std::shared_ptr<Transform>()>;   


class TransformFactory {  
   public:
      std::shared_ptr<Transform> getClassByName(std::string className);
      void registClass(std::string name, CreateTransform_Func method);	  
      static TransformFactory& getInstance();
   private:
      std::map<std::string, CreateTransform_Func> m_classMap;
      std::mutex mtx; 
      TransformFactory()=default; 
};


class RegisterTransform{
   public:
      RegisterTransform(std::string className,CreateTransform_Func ptrCreateFn){
         TransformFactory::getInstance().registClass(className,ptrCreateFn);
      }
};

  

#define REGISTERTRANSFORM(regiterName,className)               \
   static RegisterTransform g_creatorRegister##regiterName(    \
      #regiterName, []()->std::shared_ptr<Transform>{          \
         return std::make_shared<className>();                 \
      });




}



#endif