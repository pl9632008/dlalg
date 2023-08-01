#include "dl_algfactory.h"
#include "dl_common.h"



namespace yjh_deeplearning{


int BaseDLAlg::ProcessPic(const std::vector<AIInputInfo> &input_list,std::vector<AIOutputInfo> &output_list)
{
	//not implemented
	return DLFAILED;
}


int BaseDLAlg::DeInit()
{
    //not implemented
	return DLFAILED;
}

DLAlgClassFactory& DLAlgClassFactory::getInstance() {
	static DLAlgClassFactory sLo_factory;
	return sLo_factory;
}

std::shared_ptr<BaseDLAlg> DLAlgClassFactory::getClassByName(std::string className) {
	auto iter = m_classMap.find(className);
	if (iter == m_classMap.end())
		return nullptr;
	else
		return iter->second();
}

void DLAlgClassFactory::registClass(std::string name, CreateDLAlg_Func method) {
	mtx.lock();
	m_classMap.insert(make_pair(name, method));
	mtx.unlock();
}


}