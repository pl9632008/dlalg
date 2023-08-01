/******************************************************************
*  Copyright(c) 2015-2025 yijiahe
*  All rights reserved.
*
*  description:yijiahe face recognition algorithm interface
*
*  version: 1.0
*  author: dongbangfa
*  date: 2022.10.12
******************************************************************/
#ifndef DL_FACE_RECOGNITION_H
#define DL_FACE_RECOGNITION_H
#include <vector>
#include <memory>
#include <string>


namespace yjh_deeplearning{


/**
	* @brief 深度学习算法接口
	* 每个实例对应一个算法
*/
class FaceRecognitionAlgorithm{
	public:
		/**
		* @brief 初始化函数。
		* @param alg_conf_file 入参 算法配置文件
		* @param gpu_index 入参 默认值是-1，表示不用gpu
		* @return 返回(0:成功; 其他参考错误码文档)
		*/
		int Init(std::string alg_conf_file,int gpu_index = -1);

		
		/**
		* @brief 图片处理函数
		* @param ori_img 入参 传入图片内存指针
        * @param features 出参，返回图片人脸特征值
		* @return 返回(0:成功;  其他参考错误码文档)
		*/
		int ProcessImage(const std::shared_ptr<void> &ori_img,std::vector<float> &features);

		/**
		* @brief 图片处理函数
		* @param ori_img 入参 传入图片内存指针
		* @param face_det 出参，返回图片人脸框坐标，元素个数4个，依次是中心点横纵坐标和宽高值
        * @param features 出参，返回图片人脸特征值
		* @return 返回(0:成功;  其他参考错误码文档)
		*/
		int ProcessImage(const std::shared_ptr<void> &ori_img,std::vector<int> &face_det,std::vector<float> &features);


        /**
		* @brief 人脸特征对比函数
		* @param face_1 入参 人脸1的特征值
        * @param face_2 入参 人脸2的特征值
        * @param threshold 入参 相似度比较阈值，低于该阈值的返回false，表示不是同一人，返回true表示是同一个人	
		* @return 返回(true同一个人，false，不是同一个人)
		*/
        bool IsSamePeople(std::vector<float> &face_1,std::vector<float> &face_2,float threshold=0.4);

	private:
		class FaceRecognitionAlgorithmImp;
		std::shared_ptr<FaceRecognitionAlgorithmImp> face_alg_imp_;	

};

}

extern "C"  {

	/**
		* @brief 实例化人脸识别类。	
		* @return 返回人脸识别类
		*/
	yjh_deeplearning::FaceRecognitionAlgorithm *GetFaceRecognitionAlgorithmInstance();
	
	/**
		* @brief 销毁人脸识别类，释放内存。	
		* @param self 入参 人脸识别类
		*/
	void DestoryFaceRecognitionAlgorithmInstance(yjh_deeplearning::FaceRecognitionAlgorithm *facepointer);	

	/**
		* @brief 初始化函数。
		* @param self 入参 人脸识别类
		* @param alg_conf_file 入参 算法配置文件
		* @param gpu_index 入参 默认值是-1，表示不用gpu
		* @return 返回(0:成功; 其他参考错误码文档)
		*/
	int InitFaceRecognitionAlgorithm(yjh_deeplearning::FaceRecognitionAlgorithm *self,char *alg_conf_file,int gpu_index = -1);
	
	/**
		* @brief 网络编码图片处理函数
		* @param self 入参 人脸识别类
		* @param img_ptr 入参 传入base64编码的图片的指针
        * @param img_length 入参 传入图片编码的长度 
		* @param features 出参，输出算法分析后人脸特征值，用户需要提前分配好内存，建议分配为512
		* @param size 出入参，入参时表示用户分配的features内存大小，如果分配的大小小于人脸特征值的维度，会返回错误，出参表示算法分析后人脸特征值的维度
		* @return 返回(0:成功;  其他参考错误码文档)
		*/
	int ProcessFaceImage(yjh_deeplearning::FaceRecognitionAlgorithm *self,char *img_ptr,int img_length,float *features,int *size);

		/**
		* @brief 本地图片处理函数
		* @param self 入参 人脸识别类
		* @param img_ptr 入参 传入本地图片路径
        * @param cneter_x 出参 检测到人脸框中心横坐标地址
		* @param center_y 出参 检测到人脸框中心纵坐标地址
		* @param width 出参 检测到人脸框宽度
		* @param height 出参 检测到人脸框高度
		* @param features 出参，输出算法分析后人脸特征值，用户需要提前分配好内存，建议分配为512
		* @param size 出入参，入参时表示用户分配的features内存大小，如果分配的大小小于人脸特征值的维度，会返回错误，出参表示算法分析后人脸特征值的维度
		* @return 返回(0:成功;  其他参考错误码文档)
		*/
	int ProcessFaceLocalImage(yjh_deeplearning::FaceRecognitionAlgorithm *self,char *img_ptr,int *center_x,int *center_y,int *width,int *height,float *features,int *size);

	/**
	* @brief 人脸特征对比函数
	* @param self 入参 人脸识别类
	* @param face_1 入参 人脸1的特征值
	* @param size_1 入参 人脸1的特征值的维度
	* @param face_2 入参 人脸2的特征值
	* @param size_2 入参 人脸2的特征值的维度
	* @param threshold 入参 相似度比较阈值，低于该阈值的返回false，表示不是同一人，返回true表示是同一个人	
	* @return 返回(true同一个人，false，不是同一个人)
	*/	
	bool IsSamePeople(yjh_deeplearning::FaceRecognitionAlgorithm *self,float *face_1,int size_1,float *face_2,int size_2,float threshold=0.4);	
}

#endif
