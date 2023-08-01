#include "dl_util.h"
#include "dl_common.h"
#include "clipper.hpp"

#include <glog/logging.h>


namespace yjh_deeplearning{

//标准的人脸关键点。
float src_landmark[5][2] = {
    {30.2946f, 51.6963f},
    {65.5318f, 51.5014f},
    {48.0252f, 71.7366f},
    {33.5493f, 92.3655f},
    {62.7299f, 92.2041f}};

cv::Mat meanAxis0(const cv::Mat &src)
{
    int num = src.rows;
    int dim = src.cols;

    // x1 y1
    // x2 y2

    cv::Mat output(1,dim,CV_32F);
    for(int i = 0 ; i <  dim; i ++)
    {
        float sum = 0 ;
        for(int j = 0 ; j < num ; j++)
        {
            sum+=src.at<float>(j,i);
        }
        output.at<float>(0,i) = sum/num;
    }

    return output;
}
 
cv::Mat elementwiseMinus(const cv::Mat &A,const cv::Mat &B)
{
    cv::Mat output(A.rows,A.cols,A.type());

    assert(B.cols == A.cols);
    if(B.cols == A.cols)
    {
        for(int i = 0 ; i <  A.rows; i ++)
        {
            for(int j = 0 ; j < B.cols; j++)
            {
                output.at<float>(i,j) = A.at<float>(i,j) - B.at<float>(0,j);
            }
        }
    }
    return output;
}

 
cv::Mat varAxis0(const cv::Mat &src)
{
    cv::Mat temp_ = elementwiseMinus(src,meanAxis0(src));
    cv::multiply(temp_ ,temp_ ,temp_ );
    return meanAxis0(temp_);

}



int MatrixRank(cv::Mat M)
{
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = countNonZero(nonZeroSingularValues);
    return rank;

}

//    References
//    ----------
//    .. [1] "Least-squares estimation of transformation parameters between two
//    point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
//
//    """
//
//    Anthor:Jack Yu
    cv::Mat similarTransform(cv::Mat src,cv::Mat dst) {
        int num = src.rows;
        int dim = src.cols;
        cv::Mat src_mean = meanAxis0(src);
        cv::Mat dst_mean = meanAxis0(dst);
        cv::Mat src_demean = elementwiseMinus(src, src_mean);
        cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
        cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
        cv::Mat d(dim, 1, CV_32F);
        d.setTo(1.0f);
        if (cv::determinant(A) < 0) {
            d.at<float>(dim - 1, 0) = -1;
 
        }
        cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
        cv::Mat U, S, V;
        cv::SVD::compute(A, S,U, V);
 
        // the SVD function in opencv differ from scipy .
 
 
        int rank = MatrixRank(A);
        if (rank == 0) {
            assert(rank == 0);
 
        } else if (rank == dim - 1) {
            if (cv::determinant(U) * cv::determinant(V) > 0) {
                T.rowRange(0, dim).colRange(0, dim) = U * V;
            } else {
//            s = d[dim - 1]
//            d[dim - 1] = -1
//            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
//            d[dim - 1] = s
                int s = d.at<float>(dim - 1, 0) = -1;
                d.at<float>(dim - 1, 0) = -1;
 
                T.rowRange(0, dim).colRange(0, dim) = U * V;
                cv::Mat diag_ = cv::Mat::diag(d);
                cv::Mat twp = diag_*V; //np.dot(np.diag(d), V.T)
                cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
                cv::Mat C = B.diag(0);
                T.rowRange(0, dim).colRange(0, dim) = U* twp;
                d.at<float>(dim - 1, 0) = s;
            }
        }
        else{
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_*V.t(); //np.dot(np.diag(d), V.T)
            cv::Mat res = U* twp; // U
            T.rowRange(0, dim).colRange(0, dim) = -U.t()* twp;
        }
        cv::Mat var_ = varAxis0(src_demean);
        float val = cv::sum(var_).val[0];
        cv::Mat res;
        cv::multiply(d,S,res);
        float scale =  1.0/val*cv::sum(res).val[0];
        T.rowRange(0, dim).colRange(0, dim) = - T.rowRange(0, dim).colRange(0, dim).t();
        cv::Mat  temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
        cv::Mat  temp2 = src_mean.t(); //src_mean.T
        cv::Mat  temp3 = temp1*temp2; // np.dot(T[:dim, :dim], src_mean.T)
        cv::Mat temp4 = scale*temp3;
        T.rowRange(0, dim).colRange(dim, dim+1)=  -(temp4 - dst_mean.t()) ;
        T.rowRange(0, dim).colRange(0, dim) *= scale;
        return T;
    }





std::vector<cv::Point> getMinBoxes(const std::vector<cv::Point> &inVec, float &minSideLen, float &allEdgeSize) {
    std::vector<cv::Point> minBoxVec;
    cv::RotatedRect textRect = cv::minAreaRect(inVec);
    cv::Mat boxPoints2f;
    cv::boxPoints(textRect, boxPoints2f);

    float *p1 = reinterpret_cast<float *>(boxPoints2f.data);
    std::vector<cv::Point> tmpVec;
    for (int i = 0; i < 4; ++i, p1 += 2) {
        tmpVec.emplace_back(int(p1[0]), int(p1[1]));
    }

    const auto& cvPointCompare= [](const cv::Point &a, const cv::Point &b) {
        return a.x < b.x;
    };
    std::sort(tmpVec.begin(), tmpVec.end(), cvPointCompare);

    int index1, index2, index3, index4;
    if (tmpVec[1].y > tmpVec[0].y) {
        index1 = 0;
        index4 = 1;
    } else {
        index1 = 1;
        index4 = 0;
    }

    if (tmpVec[3].y > tmpVec[2].y) {
        index2 = 2;
        index3 = 3;
    } else {
        index2 = 3;
        index3 = 2;
    }

    minBoxVec.push_back(tmpVec[index1]);
    minBoxVec.push_back(tmpVec[index2]);
    minBoxVec.push_back(tmpVec[index3]);
    minBoxVec.push_back(tmpVec[index4]);

    minSideLen = (std::min)(textRect.size.width, textRect.size.height);
    allEdgeSize = 2.f * (textRect.size.width + textRect.size.height);

    return minBoxVec;
}

float boxScoreFast(const cv::Mat &inMat, const std::vector<cv::Point> &inBox) {
    std::vector<cv::Point> box = inBox;
    int width = inMat.cols;
    int height = inMat.rows;
    int maxX = -INFINITY, minX = INFINITY, maxY = -INFINITY, minY = INFINITY;
    for (int i = 0; i < box.size(); ++i) {
        if (maxX < box[i].x)
            maxX = box[i].x;
        if (minX > box[i].x)
            minX = box[i].x;
        if (maxY < box[i].y)
            maxY = box[i].y;
        if (minY > box[i].y)
            minY = box[i].y;
    }
    maxX = std::min(std::max(maxX, 0), width - 1);
    minX = std::max(std::min(minX, width - 1), 0);
    maxY = std::min(std::max(maxY, 0), height - 1);
    minY = std::max(std::min(minY, height - 1), 0);

    for (int i = 0; i < box.size(); ++i) {
        box[i].x = box[i].x - minX;
        box[i].y = box[i].y - minY;
    }

    std::vector<std::vector<cv::Point>> maskBox;
    maskBox.push_back(box);
    cv::Mat maskMat(maxY - minY + 1, maxX - minX + 1, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::fillPoly(maskMat, maskBox, cv::Scalar(1, 1, 1), 1);

    return cv::mean(inMat(cv::Rect(cv::Point(minX, minY), cv::Point(maxX + 1, maxY + 1))).clone(),
                    maskMat).val[0];
}

std::vector<cv::Point> unClip(const std::vector<cv::Point> &inBox, float perimeter, float unClipRatio) {
    std::vector<cv::Point> outBox;

    ClipperLib::Path poly;

    for (int i = 0; i < inBox.size(); ++i) {
        poly.push_back(ClipperLib::IntPoint(inBox[i].x, inBox[i].y));
    }

    double distance = unClipRatio * ClipperLib::Area(poly) / (double) perimeter;

    ClipperLib::ClipperOffset clipperOffset;
    clipperOffset.AddPath(poly, ClipperLib::JoinType::jtRound, ClipperLib::EndType::etClosedPolygon);
    ClipperLib::Paths polys;
    polys.push_back(poly);
    clipperOffset.Execute(polys, distance);
    
    outBox.clear();
    std::vector<cv::Point> rsVec;
    for (int i = 0; i < polys.size(); ++i) {
        ClipperLib::Path tmpPoly = polys[i];
        for (int j = 0; j < tmpPoly.size(); ++j) {
            outBox.emplace_back(tmpPoly[j].X, tmpPoly[j].Y);
        }
    }

    return outBox;
}



int AdjustTargetImg(cv::Mat &src, int dstWidth, int dstHeight) {
    cv::Mat srcResize;
    float scale = (float) dstHeight / (float) src.rows;
    int angleWidth = int((float) src.cols * scale);
    try{
        cv::resize(src, srcResize, cv::Size(angleWidth, dstHeight));
        cv::Mat srcFit = cv::Mat(dstHeight, dstWidth, CV_8UC3, cv::Scalar(255, 255, 255));
        if (angleWidth < dstWidth) {
            cv::Rect rect(0, 0, srcResize.cols, srcResize.rows);
            srcResize.copyTo(srcFit(rect));
        } else {
            cv::Rect rect(0, 0, dstWidth, dstHeight);
            srcResize(rect).copyTo(srcFit);
        }
        src = srcFit;
    }
    catch (cv::Exception &e)
    {
        // output exception information
        LOG(ERROR) << "message: " << e.what();
        return YJH_AI_UTIL_ERROR;
    }
    
    return DLSUCCESSED;
}



cv::Mat GetRotateCropImage(const cv::Mat &src,const std::vector<cv::Point>& box) {
    cv::Mat image;
    src.copyTo(image);
    std::vector<cv::Point> points = box;

    int collectX[4] = {box[0].x, box[1].x, box[2].x, box[3].x};
    int collectY[4] = {box[0].y, box[1].y, box[2].y, box[3].y};
    int left = int(*std::min_element(collectX, collectX + 4));
    int right = int(*std::max_element(collectX, collectX + 4));
    int top = int(*std::min_element(collectY, collectY + 4));
    int bottom = int(*std::max_element(collectY, collectY + 4));

    cv::Mat imgCrop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(imgCrop);

    for (int i = 0; i < points.size(); i++) {
        points[i].x -= left;
        points[i].y -= top;
    }

    int imgCropWidth = int(sqrt(pow(points[0].x - points[1].x, 2) +
                                pow(points[0].y - points[1].y, 2)));
    int imgCropHeight = int(sqrt(pow(points[0].x - points[3].x, 2) +
                                 pow(points[0].y - points[3].y, 2)));

    cv::Point2f ptsDst[4];
    ptsDst[0] = cv::Point2f(0., 0.);
    ptsDst[1] = cv::Point2f(imgCropWidth, 0.);
    ptsDst[2] = cv::Point2f(imgCropWidth, imgCropHeight);
    ptsDst[3] = cv::Point2f(0.f, imgCropHeight);

    cv::Point2f ptsSrc[4];
    ptsSrc[0] = cv::Point2f(points[0].x, points[0].y);
    ptsSrc[1] = cv::Point2f(points[1].x, points[1].y);
    ptsSrc[2] = cv::Point2f(points[2].x, points[2].y);
    ptsSrc[3] = cv::Point2f(points[3].x, points[3].y);

    cv::Mat M = cv::getPerspectiveTransform(ptsSrc, ptsDst);

    cv::Mat partImg;
    cv::warpPerspective(imgCrop, partImg, M,
                        cv::Size(imgCropWidth, imgCropHeight),
                        cv::BORDER_REPLICATE);

    if (float(partImg.rows) >= float(partImg.cols) * 1.5) {
        cv::Mat srcCopy = cv::Mat(partImg.rows, partImg.cols, partImg.depth());
        cv::transpose(partImg, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    } else {
        return partImg;
    }
}

cv::Mat GetRoiExpandImg(cv::Mat img,DetectionObj &bbox,int target_h,int target_w,float expanding,cv::Mat &inverse_warp_mat)
{
    cv::Mat re_img;
    float aspect_ratio = target_w*1.0/target_h;
    float x = bbox.center_x;
    float y = bbox.center_y;
    float w = bbox.width;
    float h = bbox.height;
    if (w > aspect_ratio * h)
        h = w * 1.0 / aspect_ratio;
    else if (w < aspect_ratio * h)
        w = h * aspect_ratio;
    w = w*expanding;
    h = h*expanding;

    
    cv::Point2f srcTri[3];
    cv::Point2f dstTri[3];
    cv::Point2f tmpTri;
    
    srcTri[0] = cv::Point2f(bbox.center_x, bbox.center_y);
	srcTri[1] = cv::Point2f(bbox.center_x, bbox.center_y-w/2);
    tmpTri = srcTri[0] - srcTri[1];    
	srcTri[2] = srcTri[1] + cv::Point2f(-tmpTri.y, tmpTri.x);
 
	dstTri[0] = cv::Point2f(target_w*1.0/2, target_h*1.0/2);
	dstTri[1] = cv::Point2f(target_w*1.0/2, target_h*1.0/2-target_w*1.0/2);
    tmpTri = dstTri[0] - dstTri[1]; 
	dstTri[2] = dstTri[1] + cv::Point2f(-tmpTri.y, tmpTri.x);
 
	/// 求得仿射变换
	cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
    inverse_warp_mat = cv::getAffineTransform(dstTri, srcTri);
    cv::Mat imageROI;

    cv::warpAffine(img, imageROI, warp_mat, {target_w, target_h});
    return imageROI;
}

bool JudgeVecAndPolygonIntersect(const cv::Point &center,const cv::Point &head,const std::vector<cv::Point> &polygon)
{
    if(polygon.size() == 0)
    {
        return false;
    }
    cv::Point ori_vec = head - center;
    cv::Point polygon_vec;
    polygon_vec = polygon[0] - center;
    long first_cross_product = polygon_vec.x*ori_vec.y-polygon_vec.y*ori_vec.x;
    
    if(first_cross_product == 0)
    {
        return true;
    }
    for(unsigned int i=1;i<polygon.size();i++)
    {
        polygon_vec = polygon[i] - center;
        if((polygon_vec.x*ori_vec.y-polygon_vec.y*ori_vec.x)*first_cross_product <= 0)
        {           
            return true;
        }

    }

    return false;
}

float GetDistance(const cv::Point &pointO,const cv::Point &pointA)
{
    float distance;
    distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
    distance = sqrtf(distance);
    return distance;
}


float GetAngle(cv::Point pt0, cv::Point pt1, cv::Point pt2)
{
    double dx1 = (pt1.x - pt0.x);
    double dy1 = (pt1.y - pt0.y);
    double dx2 = (pt2.x - pt0.x);
    double dy2 = (pt2.y - pt0.y);
    double angle_line = (dx1*dx2 + dy1 * dy2) / sqrt((dx1*dx1 + dy1 * dy1)*(dx2*dx2 + dy2 * dy2) + 1e-10);
    double a = acos(angle_line) * 180 / 3.141592653;

    return a;
}



}