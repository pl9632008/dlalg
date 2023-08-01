/******************************************************************
*  Copyright(c) 2015-2025 yijiahe
*  All rights reserved.
*
*  description:yijiahe deep learning algorithm interface
*
*  version: 1.0
*  author: dongbangfa
*  date: 2022.01.26
******************************************************************/
#ifndef DL_ALGORITHM_H
#define DL_ALGORITHM_H
#include <vector>
#include <memory>
#include <string>

namespace yjh_deeplearning{


/**
	* @brief 深度学习算法推理入参结构体	
	* 	
**/
struct AIInputInfo{
	//需要处理的输入资源对应的智能指针列表， 使用opencv的mat进行传值。可以使用shared_ptr<cv::Mat> 直接赋值,vector表示算法模型的输入可能是多个,这个多个输入指的是模型推理的输入参数数量，不是batch size
	std::vector<std::shared_ptr<void>> src_mat{};
};


/**
	* @brief 深度学习算法推理结果结构体	
	* 	
**/
struct AIResult{	

	//分类任务：分类类别
	//检测和实例分割任务：检测框或者实例类别
	//语义分割任务:分割类别	
	std::string value;


	//分类任务：分类类别置信度
	//检测和实例分割任务：检测框类别置信度
	//相似度检测任务:相似度置信度	
	float score;            

	
	int center_x;		//目标框中心x坐标，宽度方向	
	int center_y;		//目标框中心y坐标，高度方向
	int width;  //目标框宽度
	int height;	//目标框高度
	float angle;    //旋转目标框角度

	//实例分割,语义分割, 超分，去噪，去模糊,特征值输出结果智能指针列表，使用opencv的mat进行传值。使用时需要std::static_pointer_cast<cv::Mat>进行转换，vector表示可能有多个输出
	std::shared_ptr<void> dst_mat{nullptr};	

	//关键点检测任务，一组关键点坐标
	std::vector<std::pair<int,int>> key_points{};

	int check_size; //结构体校验值
	
};

/**
	* @brief 深度学习算法推理出参结构体	
	* 	
**/
struct AIOutputInfo{ 
	
	//算法推理输出结果
	//单分类任务和相似度检测任务时，输出元素个数是1，其余任务元素个数不定。
	std::vector<AIResult> result_list{};
};



/**
	* @brief 深度学习算法推理接口  c++风格
	* 每个实例对应一个算法
*/
class AIAlgorithm{
	public:
		/**
		* @brief 初始化函数。
		* @param alg_conf_file 入参 算法配置文件
		* @param gpu_index 入参 默认值是-1，表示不用gpu
		* @return 返回(0:成功; 其他参考错误码文档)
		*/
		int Init(const std::string &alg_conf_file,int gpu_index = -1);

		
		/**
		* @brief 推理处理函数
		* @param input_info 入参 传入的需要处理相关信息
		* @param output_info 出参，输出算法分析后相关信息
		* @return 返回(0:成功;  其他参考错误码文档)
		*/
		int Inference(const AIInputInfo &input_info,AIOutputInfo &output_info);

		/**
		* @brief 推理批处理函数
		* @param input_list 入参 传入的需要处理相关信息
		* @param output_list 出参，输出算法分析后相关信息
		* @return 返回(0:成功;  其他参考错误码文档)
		*/
		int Inference(const std::vector<AIInputInfo> &input_list,std::vector<AIOutputInfo> &output_list);

	private:
		class AIAlgorithmImp;
		std::shared_ptr<AIAlgorithmImp> alg_imp;	

};


// c风格接口	
	
extern "C"  {

    /**
		* @brief 获取算法实例函数。
		* @return 返回nullptr失败
		*/
    std::shared_ptr<void> GetAIAlgorithmInstance();

    /**
		* @brief 初始化函数。
    	* @param instance 入参 算法实例
		* @param alg_conf_file 入参 算法配置文件
		* @param gpu_index 入参 默认值是-1，表示不用gpu
		* @return 返回(0:成功; 其他参考错误码文档)
		*/
    int InitAIAlgorithm(const std::shared_ptr<void> instance,const std::string &alg_conf_file,int gpu_index = -1);


     /**
		* @brief 单张推理接口。
    	* @param instance 入参 算法实例
		* @param input_info 入参 算法输入结构体
		* @param output_info 出参 算法输出结构体，
		* @return 返回(0:成功; 其他参考错误码文档)
		*/
    int AIAlgorithmInference(const std::shared_ptr<void> instance,const AIInputInfo &input_info,AIOutputInfo &output_info);

    

    /**
		* @brief 批推理接口。
    	* @param instance 入参 算法实例
		* @param inputs_info 入参 算法输入结构体 
		* @param outputs_info 出参 ，算法输出结构体
		* @return 返回(0:成功; 其他参考错误码文档)
		*/
    int AIAlgorithmBatchInference(const std::shared_ptr<void> instance,const std::vector<AIInputInfo> &inputs_info,std::vector<AIOutputInfo> &outputs_info);

}


}


#endif
