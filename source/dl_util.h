#ifndef DL_ALG_UTIL_H
#define DL_ALG_UTIL_H



#include <opencv2/opencv.hpp>
#include <memory>
#include "dl_common.h"

namespace yjh_deeplearning{

    // center_x,center_y
    struct DetectionObj
    {
            float center_x, center_y;
            float width, height;
            float angle;
            int class_idx;
            float score;
    };

    struct TextBox {
        std::vector<cv::Point> box_point_;
        float score_;
    };

    cv::Mat meanAxis0(const cv::Mat &src);    
    cv::Mat elementwiseMinus(const cv::Mat &A,const cv::Mat &B); 
    cv::Mat varAxis0(const cv::Mat &src);
    int MatrixRank(cv::Mat M);   
    cv::Mat similarTransform(cv::Mat src,cv::Mat dst);

    std::vector<cv::Point> getMinBoxes(const std::vector<cv::Point> &inVec, float &minSideLen, float &allEdgeSize);
    float boxScoreFast(const cv::Mat &inMat, const std::vector<cv::Point> &inBox);
    std::vector<cv::Point> unClip(const std::vector<cv::Point> &inBox, float perimeter, float unClipRatio);

    int AdjustTargetImg(cv::Mat &src, int dstWidth, int dstHeight);

    cv::Mat GetRotateCropImage(const cv::Mat &src,const std::vector<cv::Point>& box);
    cv::Mat GetRoiExpandImg(cv::Mat img,DetectionObj &bbox,int target_h,int target_w,float expanding,cv::Mat &inverse_warp_mat);

    bool JudgeVecAndPolygonIntersect(const cv::Point &center,const cv::Point &head,const std::vector<cv::Point> &polygon);

    
    float GetAngle(cv::Point pt0, cv::Point pt1, cv::Point pt2);
    float GetDistance(const cv::Point &pointO,const cv::Point &pointA);

}

#endif
