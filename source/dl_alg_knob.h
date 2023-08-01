#ifndef DL_ALG_KNOB_H
#define DL_ALG_KNOB_H

// 旋钮状态识别

#include "dl_model.h"
#include "dl_algfactory.h"
#include "dl_crnn_model.h"
#include "dl_dbnet_model.h"
#include "dl_yolov5_model.h"
#include "dl_litehrnet_model.h"

#include <opencv2/opencv.hpp>
#include <memory>

namespace yjh_deeplearning{





class KnobDLAlg :public BaseDLAlg {
	public:      
		int Init(dlalg_jsons::AlgInfo &algInfo);
		int ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info);     
		int DeInit();
	private:
     
        int GetKnobPosition(const std::vector<cv::Point> &kepointVec,const std::vector<TextBox> &text_boxes,std::vector<int> &konb_index);

		int GetIntersectLabel(const std::vector<std::string> &str_vec,const std::vector<TextBox> &text_boxes, std::string &retrun_str);
		int GetUnIntersectLabel(const std::vector<std::string> &str_vec,const std::vector<TextBox> &text_boxes,const std::vector<cv::Point> &kepoint_vec,std::string &retrun_str,float &score);

		int GetSplitLabel(const std::string &ocr_str,const TextBox &text_box,const std::vector<cv::Point> &kepoint_vec,std::string &label_str); //处理合并在一块的标签，例如开关
		
		inline cv::Rect GetRectByObj(const cv::Mat &img,const DetectionObj &obj);

		std::shared_ptr<Yolov5Model> yolov5_model_{nullptr};
		std::shared_ptr<LiteHrnetModel> keypoint_model_{nullptr};
        std::vector<std::string> yolov5_class_name_;

        int keypoint_height_{0};
        int keypoint_width_{0};
               
        float expanding_{1.25};
        std::vector<DetectionObj> detection_results_;
        cv::Mat inverse_warp_mat_;

		std::shared_ptr<DbnetModel> dbnet_model_{nullptr};
		std::shared_ptr<CRNNModel> crnn_model_{nullptr};

		std::vector<std::string> crnn_class_name_;
		std::vector<std::string>  other_list_;

       
        std::vector<TextBox> text_boxes_;

    
};


}

#endif
