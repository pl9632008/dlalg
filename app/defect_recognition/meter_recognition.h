#include <iostream>
#include <string>
#include <vector>
#include "yjh_platform_common.h"

/**
 * 获取SDK版本信息
 * @return SDK版本信息
 */
std::string GetMeterSDKVersion();

/**
 * 设置场景工作模式
 * @param engine   ： 模型实例引擎
 * @param mode_type： 场景工作模式
 * @return         ： 错误返回码, 成功返回0, 失败返回非0
 */
int SetWorkMode(void *engine, int mode_type);

/**
 * 加载模型
 * @param model_path： CNN模型文件夹路径
 * @param gpu_index ： 指定GUP, 为-1表示不需要绑定GPU,即使用CPU
 * @param engine    ： 返回模型实例引擎
 * @return          ： 错误返回码, 成功返回0, 失败返回非0
 */
int MeterLoadModel(const std::string &model_path, int gpu_index, void *&engine);

/**
 * 测点识别
 * @param engine       ： 模型实例引擎
 * @param image_data   ： 可见光图像数据(可见光相机或者深度相机拍摄)或者音频数据,作为图像数据时传入OpenCV的Mat格式
 * @param depth_data   ： 深度点云数据,封装成OpenCV的CV_32FC3 Mat格式,XYZ通道分别对应BGR三通道,若不需要则传入NULL
 * @param config_path  ： 模板文件夹路径
 * @param detect_result： 返回算法识别结果
 * @param output_num   ： 返回输出结果的个数
 * @return             ： 错误返回码, 成功返回0, 失败返回非0
 */
int MeterRecognition(void *engine, void *image_data, void *depth_data, const std::string &config_path,
                     std::vector<DETECT_RESULT> &detect_result);  //识别接口

/**
 * 释放模型
 * @param engine： 模型实例引擎
 * @return      ： 错误返回码, 成功返回0, 失败返回非0
 */
int MeterReleaseModel(void *&engine);
