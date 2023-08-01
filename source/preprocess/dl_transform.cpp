#include "dl_transform.h"
#include "dl_common.h"



namespace yjh_deeplearning{


TransformFactory& TransformFactory::getInstance() {
	static TransformFactory sLo_factory;
	return sLo_factory;
}

std::shared_ptr<Transform> TransformFactory::getClassByName(std::string className) {
	auto iter = m_classMap.find(className);
	if (iter == m_classMap.end())
		return nullptr;
	else
		return iter->second();
}

void TransformFactory::registClass(std::string name, CreateTransform_Func method) {
	mtx.lock();
	m_classMap.insert(make_pair(name, method));
	mtx.unlock();
}


}