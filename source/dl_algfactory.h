#ifndef DL_ALGFACTORY_H
#define DL_ALGFACTORY_H

#include "dl_format.h"
#include "dl_algorithm.h"

#include <memory>
#include <map>
#include <mutex>
#include <functional>


namespace yjh_deeplearning{

class BaseDLAlg{
   public:	   
	   BaseDLAlg()=default;
	   virtual ~BaseDLAlg()=default;
      //初始化函数，子类必须实现
	   virtual int Init(dlalg_jsons::AlgInfo &algInfo)=0;

      //算法任务处理函数，单张图片处理，子类必须实现
	   virtual int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)=0;

      //算法任务处理函数，批量图片处理，
	   virtual int ProcessPic(const std::vector<AIInputInfo> &input_list,std::vector<AIOutputInfo> &output_list);

      //资源释放函数，有些算法在所有推理任务完成后，需要释放资源,此时子类需要重新实现该接口
      virtual int DeInit();
		
   };


using CreateDLAlg_Func = std::function<std::shared_ptr<BaseDLAlg>()>;   
  
   
class DLAlgClassFactory {  
   public:
      std::shared_ptr<BaseDLAlg> getClassByName(std::string className);
      void registClass(std::string name, CreateDLAlg_Func method);	  
      static DLAlgClassFactory& getInstance();
   private:
      std::map<std::string, CreateDLAlg_Func> m_classMap;
      std::mutex mtx; 
      DLAlgClassFactory()=default; 
};


class RegisterDLAlg{
   public:
      RegisterDLAlg(std::string className,CreateDLAlg_Func ptrCreateFn){
         DLAlgClassFactory::getInstance().registClass(className,ptrCreateFn);
      }
};

  

#define REGISTERALG(regiterName,className)                 \
   static RegisterDLAlg g_creatorRegister##regiterName(    \
      #regiterName, []()->std::shared_ptr<BaseDLAlg>{      \
         return std::make_shared<className>();             \
      });


}



#endif