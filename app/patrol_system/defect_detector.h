/******************************************************************
*  Copyright(c) 2015-2025 yijiahe
*  All rights reserved.
*
*  description:yijiahe 220KV Intelligent Patrol System defect recognition algorithm interface
*
*  version: 1.0
*  author: liumin
*  date: 2023.4.14
******************************************************************/
#ifndef DEFECT_DETECTOR_H
#define DEFECT_DETECTOR_H
#include <vector>
#include <memory>
#include <string>

#include "dl_algorithm.h"


namespace yjh_deeplearning{


/**
	* @brief 深度学习算法接口
	* 每个实例对应一个算法
*/
class PatrolSystemDefectRecognition{

	public:
		/**
		* @brief 初始化函数。
		* @param alg_conf_file 入参 算法配置文件
		* @param gpu_index 入参 默认值是-1，表示不用gpu
		* @return 返回(0:成功; 其他参考错误码文档)
		*/
		int Init(std::string alg_conf_file, int gpu_index = -1);

		/**
		* @brief 图片处理函数
		* @param imgs_path 入参 传入图片内存指针
        * @param detect_result 出参，返回缺陷检测结果
		* @return 返回(0:成功;  其他参考错误码文档)
		*/
		int ProcessImage(const std::vector<std::string> &imgs_path, std::vector<AIOutputInfo> &detect_result, std::vector<std::string> &img_info);

		/**
		* @brief 图片处理函数
		* @param imgs_path 入参 传入图片内存指针
        * @param detect_json 出参，返回缺陷检测结果
		* @return 返回(0:成功;  其他参考错误码文档)
		*/
		int ProcessImageJson(const std::vector<std::string> &imgs_path, std::string &detect_json);

	private:
		class PatrolSystemDefectRecognitionImp;
		std::shared_ptr<PatrolSystemDefectRecognitionImp> defect_alg_imp_;
};


}


#endif


