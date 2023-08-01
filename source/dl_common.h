#ifndef DL_COMMON_H
#define DL_COMMON_H
#include <string>
#include <vector>
namespace yjh_deeplearning{

typedef enum AI_RETURN_VALUE_ {
    DLSUCCESSED  = 0,
    DLFAILED = -10000,
}AI_RETURN_VALUE;

const float IMAGE_MAX_VALUE=255.0;

const unsigned int  GENERAL_STDORMEAN_SIZE= 3;

const unsigned int  MAX_MEMORY_BLOCK_SIZE= 32;

const std::string INFER_ENGINE_OPENCV = "opencv";
const std::string INFER_ENGINE_TRT = "tensorrt";
const std::string INFER_ENGINE_ORT = "onnxruntime";
const std::string INFER_ENGINE_LIBTORCH = "libtorch";
const std::string INFER_ENGINE_ASCENDCL = "ascendcl";

const std::string UNKNOWN_KNOB = "unknown";


#define CHECK_SUCCESS(FUN) \
{int error_code = (FUN); \
 if (error_code != DLSUCCESSED){ \
        return error_code;}}

typedef enum AI_errorcode_e_ {
    YJH_AI_OK = 0,                      // 正常

    YJH_AI_UNKNOW_ERROR             = -1,    // 未知异常,保留使用
    YJH_AI_MEMORY_ERROR             = -2,    // 内存信息异常,任何时候发现内存异常,则使用
    YJH_AI_NO_MEMORY                = -3,    // 内存不足,内存开辟不足情况使用
    YJH_AI_INPUT_ERROR              = -4,    // 入参异常,对于函数入参问题类型很多,难以具体,一般可以用这个错误码,可以具体描述的使用下面的详细错误码
    YJH_AI_IMAGE_INFO_ERROR         = -5,    // 图像信息异常,对于入参图像,如果宽高异常,颜色空间异常,指针为空等情况使用
    YJH_AI_VIDEO_INFO_ERROR         = -6,    // 视频信息异常,对于入参视频信息,如果宽高异常,帧率异常等情况使用
    YJH_AI_RULE_INFO_ERROR          = -7,    // 功能配置信息异常,对于功能配置信息,如果存在异常则使用
    YJH_AI_NO_NETWORK_WEIGHT        = -8,    // 网络模型不存在
    YJH_AI_NETWORK_LOAD_ERROR       = -9,    // 网络加载异常
    YJH_AI_VIDEO_ERROR              = -10,
    YJH_AI_PARAMETER_ERROR          = -11,
    YJH_AI_CFGFILE_NOFIND_ERROR          = -12,  //算法配置文件打不开
    YJH_AI_CFGFILE_PARSE_ERROR          = -13,  //算法配置文件解析错误
    YJH_AI_ALG_NOFIND_ERROR          = -14,  // 算法尚未实现
    YJH_AI_ALG_INIT_ERROR          = -15,  // 算法初始化失败
    YJH_AI_INPUT_STDORMEAN_ERROR   =  -16, //配置参数错误

   
    YJH_AI_INPUT_IMG_ERROR          = -50,  // 用户传入图片为空
    YJH_AI_INPUT_IMGTYPE_ERROR          = -51,  // 用户传入图片格式错误
    YJH_AI_INPUT_IMGPREPROCESS_ERROR          = -52,  // 用户传入图片格式错误
    YJH_AI_INPUT_IMGFORMAT_ERROR          = -53,  // 用户传入图片格式错误
    


    YJH_AI_OPENCV_TYPE_ERROR          = -100,  // opencv模型类型不支持
    YJH_AI_OPENCV_INIT_ERROR          = -101,  // 算法初始化失败
    YJH_AI_OPENCV_SOFTMAX_ERROR          = -102,  // opencv softmax失败
    YJH_AI_OPENCV_INFERENCE_ERROR          = -103,  // opencv推理失败

    YJH_AI_UTIL_ERROR          = -110,  // opencv推理失败

   

    

    YJH_AI_LIBTORCH_INIT_ERROR          = -200,  // libtorch初始化失败
    YJH_AI_LIBTORCH_INFERENCE_ERROR          = -201,  // libtorch推理失败


    
    //人脸相关
    YJH_AI_FACE_NONE_ERROR          = -401,  // 照片没有人脸 
    YJH_AI_FACE_IMAGE_ERROR          = -402,  // 传入照片错误

    // //*****自动建站相关 1000-1999*****//
    // AUTOBUILD_ERROR                              = -1000,
    // AUTOBUILD_IMAGE_STITCH_ERROR                 = AUTOBUILD_ERROR-1,    // 图像拼接失败
    // AUTOBUILD_IMAGE_STITCH_NUM_ERROR             = AUTOBUILD_ERROR-2,    // 图像输入数量错误
    // AUTOBUILD_IMAGE_STITCH_LITTLE_KEYPOINT_ERROR = AUTOBUILD_ERROR-3,    // 图像特征点过少
    // AUTOBUILD_IMAGE_STITCH_OPENCV_ERROR          = AUTOBUILD_ERROR-4,    // 图像拼接opencv失败
    // AUTOBUILD_NOT_FIND_CABINET_ERROR             = AUTOBUILD_ERROR-5,    // 图像中未检测出柜体
    // AUTOBUILD_NOT_FIT_CABINET_ERROR              = AUTOBUILD_ERROR-6,    // 图像中未检测出比例合适柜体
    // AUTOBUILD_NOT_FIND_WHOLE_CABINET_ERROR       = AUTOBUILD_ERROR-7,    // 图像中未检测出完整柜体
    // AUTOBUILD_NOT_FIND_TEMPLATE_IMAGE_ERROR      = AUTOBUILD_ERROR-8,    // 模版路径中没有图片
    // AUTOBUILD_NOT_FIND_TEMPLATE_PATH_ERROR       = AUTOBUILD_ERROR-9,    // 模版路径中没有子文件夹
    // AUTOBUILD_NOT_FIND_TEMPLATE_REID_ERROR       = AUTOBUILD_ERROR-10,   // reid未能找到模版
    // AUTOBUILD_NOT_FIND_METER_NAME_REID_ERROR     = AUTOBUILD_ERROR-11,   // reid未能找到子类名称
    // AUTOBUILD_METER_ALIGN_REID_ERROR             = AUTOBUILD_ERROR-12,   // 表计配准失败

    // //*****火焰识别相关 2000-2999*****//
    // FIREDET_ERROR                                = -2000,

    // //******软件平台相关 3000-3999 ******//
    // PLATFORM_ERROR                               = -3000,
    // PLATFORM_NOT_FIND_FILE_ERROR                 = PLATFORM_ERROR-1,     // 文件找不到
    // PLATFORM_FILE_OPEN_ERROR                     = PLATFORM_ERROR-2,     // 文件打开错误
    // PLATFORM_FILE_READ_ERROR                     = PLATFORM_ERROR-3,     // 文件读取错误
    // PLATFORM_ENGINE_INFO_ERROR                   = PLATFORM_ERROR-4,     // 引擎信息异常
    // PLATFORM_FILE_EMPTY_ERROR                    = PLATFORM_ERROR-5,     // 读取文件,内容为空
    // PLATFORM_AI_TASK_TYPE_ERROR                  = PLATFORM_ERROR-6,     // 任务类型错误
    // PLATFORM_AI_TASK_MATCH_ERROR                 = PLATFORM_ERROR-7,     // 任务匹配错误
    // PLATFORM_AI_FRAMEWORK_ERROR                  = PLATFORM_ERROR-8,     // 深度学习框架错误
    // PLATFORM_NOT_SUPPORT                         = PLATFORM_ERROR-9,     // 不支持该功能,框架或模型等
    // PLATFORM_NET_OUTPUT_NUM_ERROR                = PLATFORM_ERROR-10,    // 网络输出层数量错误,如二分类只有1个输出层
    // PLATFORM_INCLUDE_EXTRA_ITEMS                 = PLATFORM_ERROR-11,    // 包含多余项,如各类别中包含多余标签


    YJH_AI_ERROR_END = -100000,
}AI_ERROR;



}

#endif