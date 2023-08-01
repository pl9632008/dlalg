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
#ifndef DEFECT_DETECTOR_C_API_H
#define DEFECT_DETECTOR_C_API_H
#include "defect_detector.h"
#include <cstring>

namespace yjh_deeplearning{


/**
	* @brief 缺陷检测算法单个结果结构体	
	* 	
**/
struct DefectResult
{
	char class_name[128];		// 检测框类别
	float score;       // 检测框类别置信度 
	int center_x;		// 目标检测任务：目标框中心x坐标，宽度方向	
	int center_y;		// 目标检测任务：目标框中心y坐标，高度方向
	int width;  		// 目标检测任务目标框宽度
	int height;		// 目标检测任务目标框高度
};

/**
	* @brief 缺陷检测算法单张图片结果结构体	
	* 	
**/
struct DefectOutput
{
	DefectOutput()
	{
		result_list = nullptr;
		result_size = 0;
		memset(img_flag, 0, 10);
	}
	DefectResult *result_list;
	int result_size;    //输出结果数量 
	char img_flag[10];
};



extern "C"  {

	/**
		* @brief 实例化识别类
		* @return 返回识别类
	*/
	yjh_deeplearning::PatrolSystemDefectRecognition *GetSBAlgorithmInstance();


	/**
	* @brief 销毁识别类, 释放内存。	
	* @param instance 入参 缺陷识别类指针
	*/
	void DestorySBAlgorithmInstance(yjh_deeplearning::PatrolSystemDefectRecognition *instance);	

	/**
	* @brief 初始化函数。
	* @param alg_conf_file 入参 算法配置文件
	* @param gpu_index 入参 默认值是-1，表示不用gpu
	* @return 返回(0:成功; 其他参考错误码文档)
	*/
	int InitSBAlgorithm(yjh_deeplearning::PatrolSystemDefectRecognition *self, char *alg_conf_file, int gpu_index = -1);
	
	/**
	* @brief 图片处理函数
	* @param img_path_list 入参 传入图片路径列表
	* @param input_size    入参 传入图片数量
	* @param output_info   出参 输出算法分析后相关信息, 目标框值含义为c_x, c_y, w, h
	* @return 返回(0:成功;  其他参考错误码文档)
	*/
	int ProcessSBImage(yjh_deeplearning::PatrolSystemDefectRecognition *self, char **img_path_list, int input_size, DefectOutput **output_infos);

	/**
	* @brief 图片处理函数
	* @param img_path_list 入参 传入图片路径列表
	* @param input_size    入参 传入图片数量
	* @param output_info   出参 json字符串格式输出算法分析后相关信息, 目标框值含义为c_x, c_y, w, h
	* @return 返回(0:成功;  其他参考错误码文档)
	*/
	int ProcessSBImageJson(yjh_deeplearning::PatrolSystemDefectRecognition *self, char **img_path_list, int input_size,const char** output_json);

	}

}

#endif


