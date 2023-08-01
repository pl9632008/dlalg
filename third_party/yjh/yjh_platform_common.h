//
// Created by lm on 2021/7/13.
//

#ifndef _YJH_PLATFORM_COMMON_H
#define _YJH_PLATFORM_COMMON_H

#include <iostream>
#include <string>
#include <vector>

enum SensorType {
    ColorfulCamera = 1,      //可见光相机
    PtzCameraLeft = 2,       //左球机
    PtzCameraRight = 3,      //右球机
    DepthCamera = 4,         //深度相机
};

enum LocatMethod {
    FeatureMatch  = 0,  //特征点配准
    TemplateMatch = 1,  //模板匹配
};

enum TargetColor {
    Red = 0,  //红色标线
    White,    //白色标线
    Blue,     //蓝色标线
};

enum Device_Type {
    Locate = 500,       //区域定位
    PointerMeter,       //通用型指针仪表
    LEDMeter,           //通用型LED表
    IndicatorLightMeter,//通用型电器指示灯
    MetalCorrosion,               //金属锈蚀
    MultiStateMeter,    //多态开关
    SwitchMeter,        //匹配型两态开关
    RespiratorMeter,    //呼吸器
    InfraredMeter,      //红外
    OilLevelMeter,      //油位计
    OilLeakage,                //渗漏油
    QRcode,             //二维码检测
    Nest,                 //鸟巢
    AirLeak,            //漏气
    Smoking,                 //吸烟
    NoHelmet,              //未穿安全帽
    NoOveralls,               //未穿工装
    UnclosedDoor,             //箱门闭合异常
    DialsDamaged,               //表计破损
                       // Todo: add more Device Type
};

// 分类结果结构体定义
typedef struct ClassifyResult {
    Device_Type id;  // ID类别
    int cls;
    float score;        // 置信度
    std::string value;  // 分类扩展字段,根据算法定义解析结果
    void *extend;       // 其他扩展字段,暂不考虑Windows平台,SDK申请的内存由调用者释放
} CLASSIFY_RESULT;

// 坐标框结构体定义
typedef struct AlgPoint {
    float x, y;
} AlgPOINT;  // 左上角和右下角点

// 检测结果结构定义
typedef struct DetectResult {
    Device_Type id;                // ID 类别
    std::vector<AlgPoint> points;  // 目标位置数组
    std::vector<double> homo_;     // 匹配单应矩阵
    float score;                   // 置信度
    std::string value;             // 检测结果,根据算法定义解析结果
    void *extend;                  // 其他扩展字段,暂不考虑Windows平台,SDK申请的内存由调用者释放
} DETECT_RESULT;

// 目标分割相关定义
typedef struct SegmentResult {
    std::vector<AlgPoint> points;  // 目标位置数组
    float score;                   // 置信度
    std::string value;             // 分割结果(可选), 根据算法定义解析结果
    void *extend;                  // 扩展字段,暂不考虑 Windows 平台,SDK 申请的内存由调用者释放
} SEGMENT_RESULT;                  // TODO

#endif