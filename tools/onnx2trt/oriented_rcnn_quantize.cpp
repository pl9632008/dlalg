#include <iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include "opencv2/opencv.hpp"
#include <fstream>
#include <sstream>
#include "cuda_runtime_api.h"
#include "calibrator.h"
#include "utils.h"
using namespace nvinfer1;
 


float MEAN_ORIENTED_RCNN[3]{123.675, 116.28, 103.53};
float STD_ORIENTED_RCNN[3]{58.395, 57.12, 57.375};
 static inline cv::Mat preprocess_ori_img(cv::Mat& img, int input_w, int input_h) {
     cv::resize(img, img, cv::Size(input_w,input_h),0,0, cv::INTER_LINEAR); 
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F);   
    cv::subtract(img, cv::Scalar(MEAN_ORIENTED_RCNN[0], MEAN_ORIENTED_RCNN[1], MEAN_ORIENTED_RCNN[2]), img);      
    cv::divide(img, cv::Scalar(STD_ORIENTED_RCNN[0], STD_ORIENTED_RCNN[1], STD_ORIENTED_RCNN[2]), img);   
    // std::cout<<temp.at<cv::Vec3f>(512,512)[1]<<std::endl;
    return img;
}

 const char* INPUT_BLOB_NAME = "input";

// 实例化记录器界面。捕获所有警告消息，但忽略信息性消息
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kINFO)
            std::cout << msg << std::endl;
    }
} logger;
 
 
void ONNX2TensorRT(const char* ONNX_file, std::string save_ngine,std::string qunantize_type,std::string cal_dir)
{
    // 1.创建构建器的实例
    initLibNvInferPlugins(&logger, "");
//     void* handle = dlopen("/dongbangfa/code/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so", RTLD_LAZY);

// if (!handle) {
//     std::cout << "Cannot open library: " << dlerror() << std::endl;
//     return;
// }
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
 
    // 2.创建网络定义
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);
 
    // 3.创建一个 ONNX 解析器来填充网络
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
 
    // 4.读取模型文件并处理任何错误
    std::cout << "ONNX_file : "<< ONNX_file << std::endl;
    parser->parseFromFile(ONNX_file, static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    std::cout << "parse successed" << std::endl;
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    
    // 5.创建一个构建配置，指定 TensorRT 应该如何优化模型
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    

       nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    // 这里有个OptProfileSelector，这个用来设置优化的参数,比如（Tensor的形状或者动态尺寸），
 
    profile->setDimensions("input",  nvinfer1::OptProfileSelector::kMIN,  nvinfer1::Dims4(1, 3, 1024, 1024));
    profile->setDimensions("input",  nvinfer1::OptProfileSelector::kOPT,  nvinfer1::Dims4(1, 3, 1024, 1024));
    profile->setDimensions("input",  nvinfer1::OptProfileSelector::kMAX,  nvinfer1::Dims4(1, 3, 1024, 1024));
 
    config->addOptimizationProfile(profile);
  
 
    // 6.设置属性来控制 TensorRT 如何优化网络
    // 设置内存池的空间
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 24));
    // 设置低精度   注释掉为FP32
    // std::cout<<"builder->platformHasFastInt8()() ： "<<builder->platformHasFastInt8()<<std::endl;

    config->setMinTimingIterations(1);
    config->setAvgTimingIterations(8);
    builder->setMaxBatchSize(1);
    if(qunantize_type == "fp16")
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else if(qunantize_type == "int8")
    {
        
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }
    // if (builder->platformHasFastFp16())
    // {
    //     config->setFlag(nvinfer1::BuilderFlag::kFP16);
    // }
    
    PreporcessFunc preprocess_func = preprocess_ori_img;
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, 1024, 1024, cal_dir.c_str(), "int8calib.table", preprocess_func,INPUT_BLOB_NAME,false);
    
   
    config->setInt8Calibrator(calibrator);
 
    // 7.指定配置后，构建引擎
    nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
 
    // 8.保存TensorRT模型
    std::ofstream p(save_ngine, std::ios::binary);
    p.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
 
    // 9.序列化引擎包含权重的必要副本，因此不再需要解析器、网络定义、构建器配置和构建器，可以安全地删除
    delete parser;
    delete network;
    delete config;
    delete builder;
 
    // 10.将引擎保存到磁盘，并且可以删除它被序列化到的缓冲区
    delete serializedModel;
}
 
 
void exportONNX(const char* ONNX_file, std::string save_ngine,std::string qunantize_type,std::string cal_dir)
{
    std::ifstream file(ONNX_file, std::ios::binary);
    if (!file.good())
    {
        std::cout << "Load ONNX file failed! No file found from:" << ONNX_file << std::endl;
        return ;
    }
 
    std::cout << "Load ONNX file from: " << ONNX_file << std::endl;
    std::cout << "Starting export ..." << std::endl;
 
    ONNX2TensorRT(ONNX_file, save_ngine,qunantize_type,cal_dir);
 
    std::cout << "Export success, saved as: " << save_ngine << std::endl;
 
}
 
 
int main(int argc, char** argv)
{
    // 输入信息
    if(argc != 4 || argc != 5)
    {
        std::cout<<"input error,please input: ./quantize_orient_rcnn <onnx_file_path> <save_name>  <qunantize_type[fp32|fp16|int8]> | [calibrator img dir]"<<std::endl;
        exit(0);
    }
    const char* ONNX_file = argv[1];
    std::string save_ngine;
    save_ngine.assign(argv[2]);
    std::string qunantize_type;
    qunantize_type.assign(argv[3]);
    if(qunantize_type == "int8" && argc !=5)
    {
        std::cout<<"parameter error,need calibrator image dir";
        return -1;
    }
    std::string cal_dir;
    cal_dir.assign(argv[4]);
    exportONNX(ONNX_file, save_ngine,qunantize_type,cal_dir);
 
    return 0;
}
 