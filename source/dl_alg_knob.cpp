#include "dl_alg_knob.h"
#include "dl_common.h"
#include "dl_util.h"
#include <glog/logging.h>


namespace yjh_deeplearning{


int KnobDLAlg::Init(dlalg_jsons::AlgInfo &algInfo)
{

    if(algInfo.model_list.size()!=4 )
    {
        LOG(ERROR)<<"knon model size error ,its size : "<<algInfo.model_list.size();
        return YJH_AI_ALG_INIT_ERROR;
    }

    if(algInfo.model_list[1].preprocess_list.size() != 0)
    {
        keypoint_height_ = algInfo.model_list[1].preprocess_list[0].img_height;
        keypoint_width_ = algInfo.model_list[1].preprocess_list[0].img_width;
    }
    else
    {
        LOG(ERROR)<<"keypoint model input size not config";
        return DLFAILED;
    }   
   
    yolov5_model_ = std::make_shared<Yolov5Model>(algInfo.model_list[0]);
    CHECK_SUCCESS(yolov5_model_->InitModel());

    keypoint_model_ = std::make_shared<LiteHrnetModel>(algInfo.model_list[1]);
    CHECK_SUCCESS(keypoint_model_->InitModel());

    yolov5_class_name_.swap(algInfo.model_list[0].class_name);
       
    dbnet_model_ = std::make_shared<DbnetModel>(algInfo.model_list[2]);
    CHECK_SUCCESS(dbnet_model_->InitModel());


    crnn_class_name_.swap(algInfo.model_list[3].class_name);
    other_list_.swap(algInfo.model_list[3].other_list);
   
    crnn_model_ = std::make_shared<CRNNModel>(algInfo.model_list[3]);
    CHECK_SUCCESS(crnn_model_->InitModel());

   
        
    return DLSUCCESSED;
}

int KnobDLAlg::GetKnobPosition(const std::vector<cv::Point> &kepointVec,const std::vector<TextBox> &text_boxes,std::vector<int> &konb_index)
{
    konb_index.clear();
    for(unsigned int i=0;i<text_boxes.size();i++)
    {
        // LOG(ERROR)<<kepointVec[0]<<" "<<kepointVec[1]<<" "<<text_boxes[i].box_point_[0]<<" "<<text_boxes[i].box_point_[1]<<" "<<text_boxes[i].box_point_[2]<<" "<<text_boxes[i].box_point_[3];
        if(JudgeVecAndPolygonIntersect(kepointVec[0],kepointVec[1],text_boxes[i].box_point_) == true)
        {
            konb_index.emplace_back(i);
        }
    }
    return DLSUCCESSED;
}

int KnobDLAlg::GetIntersectLabel(const std::vector<std::string> &str_vec,const std::vector<TextBox> &text_boxes,std::string &retrun_str)
{
    if(str_vec.size() == 1)
    {        
        retrun_str = str_vec[0];       
    }
    else if(str_vec.size() == 2)
    {
        if(text_boxes[0].box_point_[0].y > text_boxes[1].box_point_[0].y)
        {
            retrun_str = str_vec[1]+str_vec[0];
        }
        else
        {
            retrun_str = str_vec[0]+str_vec[1];
        }
    }
    else
    {
        retrun_str = UNKNOWN_KNOB;
    }

    return DLSUCCESSED;
}

int KnobDLAlg::GetUnIntersectLabel(const std::vector<std::string> &str_vec,const std::vector<TextBox> &text_boxes,const std::vector<cv::Point> &kepoint_vec,std::string &retrun_str,float &score)
{
    std::vector<unsigned int> flag_indexs{};
    std::vector<cv::Point2f> intersectingRegion;
    cv::RotatedRect rect1,rect2;
    cv::Point center_point;
    std::string temp_str;
    float  min_distance = std::numeric_limits<float>::max();
    for(unsigned int i=0;i<text_boxes.size();i++)
    {
        if(std::find(flag_indexs.begin(),flag_indexs.end(),i) != flag_indexs.end())
        {
            continue;
        }
        rect1 =  cv::minAreaRect(text_boxes[i].box_point_);
        unsigned int j;
        for(j=i+1;j<text_boxes.size();j++)
        {
            intersectingRegion.clear();
            rect2 = cv::minAreaRect(text_boxes[j].box_point_);
            cv::rotatedRectangleIntersection(rect1, rect2, intersectingRegion);
            if(!intersectingRegion.empty())
            {
                flag_indexs.emplace_back(j);
                break;
            }
        }
        
        if(j != text_boxes.size())
        {
            center_point.x = (rect1.center.x+rect2.center.x)/2;
            center_point.y = (rect1.center.y+rect2.center.y)/2;            
        }
        else
        {
            center_point = rect1.center;
        }        
        if(GetDistance(center_point,kepoint_vec[1]) < min_distance)
        {
            min_distance = GetDistance(center_point,kepoint_vec[1]);
            if(j != text_boxes.size())
            {
                if(rect1.center.y > rect2.center.y)
                {
                    retrun_str = str_vec[j] + str_vec[i];
                }
                else
                {
                    retrun_str = str_vec[i] + str_vec[j];
                }
            }
            else
            {
                retrun_str = str_vec[i];
            } 
            score = text_boxes[i].score_;
        }
    }    

    return DLSUCCESSED;
}

int KnobDLAlg::GetSplitLabel(const std::string &ocr_str,const TextBox &text_box,const std::vector<cv::Point> &kepoint_vec,std::string &label_str)
{
    cv::Point top_center = cv::Point((text_box.box_point_[0].x+text_box.box_point_[1].x)/2,(text_box.box_point_[0].y+text_box.box_point_[1].y)/2);
    cv::Point bottom_center = cv::Point((text_box.box_point_[0].x+text_box.box_point_[1].x)/2,(text_box.box_point_[0].y+text_box.box_point_[1].y)/2);

    cv::RotatedRect left_rotate_rec = cv::minAreaRect(std::vector<cv::Point>({text_box.box_point_[0],top_center,bottom_center,text_box.box_point_[3]}));
    cv::RotatedRect right_rotate_rec = cv::minAreaRect(std::vector<cv::Point>({top_center,text_box.box_point_[1],text_box.box_point_[2],bottom_center}));

    std::string left_str,right_str;
    int start_index = 0;
    left_str = ocr_str.substr(start_index,ocr_str.size()/2);
    if(ocr_str.size()%2 == 0)
    {
        start_index = ocr_str.size()/2;
    }
    else
    {
        start_index = ocr_str.size()/2+1;
    }
    right_str = ocr_str.substr(start_index,ocr_str.size()/2);
   
    if(GetAngle(kepoint_vec[0],kepoint_vec[1],left_rotate_rec.center)<GetAngle(kepoint_vec[0],kepoint_vec[1],right_rotate_rec.center))
    {
        label_str = left_str;
    }
    else
    {
        label_str = right_str;
    }
    return DLSUCCESSED;
}

cv::Rect KnobDLAlg::GetRectByObj(const cv::Mat &img,const DetectionObj &obj)
{
    
    int x = std::max(0,int(obj.center_x-obj.width/2));
    int y = std::max(0,int(obj.center_y-obj.height/2));
    int width =  std::min(int(obj.width),img.cols-x);
    int height = std::min(int(obj.height),img.rows-y);
    return cv::Rect(x,y,width,height);
    


}

int KnobDLAlg::ProcessPic(const AIInputInfo &input_info,AIOutputInfo &output_info)
{    
    
    if(input_info.src_mat.size() ==0 || input_info.src_mat[0] == nullptr )
    {
        return YJH_AI_INPUT_IMG_ERROR;
    }
    std::shared_ptr<cv::Mat> ori_img = std::static_pointer_cast<cv::Mat>(input_info.src_mat[0]);    
    
    CHECK_SUCCESS(yolov5_model_->ModelPredict(*ori_img,detection_results_));
    output_info.result_list.clear();
    AIResult result;   
    std::vector<cv::Point> pointVec; 
    std::vector<cv::Point> kepointVec;
    for(unsigned int i=0;i<detection_results_.size();i++)
    {
        if(detection_results_[i].class_idx<yolov5_class_name_.size())
        {  
            result.center_x = detection_results_[i].center_x;
            result.center_y = detection_results_[i].center_y;
            result.width = detection_results_[i].width;
            result.height = detection_results_[i].height;
            cv::Mat rot_imt = GetRoiExpandImg(*ori_img,detection_results_[i],keypoint_height_,keypoint_width_,expanding_,inverse_warp_mat_);
           
            keypoint_model_->ModelPredict(rot_imt,pointVec);
            
        
            cv::Point p;
            kepointVec.clear();
            for(unsigned int j=0;j<pointVec.size();j++)
            {                
                p.x = inverse_warp_mat_.ptr<double>(0)[0] * pointVec[j].x + inverse_warp_mat_.ptr<double>(0)[1] * pointVec[j].y + inverse_warp_mat_.ptr<double>(0)[2];
                p.y = inverse_warp_mat_.ptr<double>(1)[0] * pointVec[j].x + inverse_warp_mat_.ptr<double>(1)[1] * pointVec[j].y + inverse_warp_mat_.ptr<double>(1)[2];
                kepointVec.emplace_back(p);
            }
            if(kepointVec.size() != 3)
            {
                continue;
            }

            // 获取旋钮检测区域
            cv::Rect rect= GetRectByObj(*ori_img,detection_results_[i]);             
            cv::Mat text_mat = (*ori_img)(rect);

            cv::imwrite("text_mat.jpg",text_mat);           
            //关键点坐标位置偏移
            for(unsigned int k = 0;k<kepointVec.size();k++) 
            {
                kepointVec[k].x = kepointVec[k].x-rect.x;
                kepointVec[k].y = kepointVec[k].y-rect.y;
            }
            
            //文字检测
            CHECK_SUCCESS(dbnet_model_->ModelPredict(text_mat,text_boxes_));

            cv::Mat roi_img;
            std::string ocr_str;
            float score;
            std::string label_str = UNKNOWN_KNOB;

            std::vector<TextBox> temp_text_boxes{};     

            std::vector<std::string> vec_str{};
            std::vector<int> ray_intersect_index{};

            //查看指针是否穿过标签
            GetKnobPosition(kepointVec,text_boxes_,ray_intersect_index);
            // LOG(ERROR)<<result.center_x<<" "<<result.center_y<<" "<<ray_intersect_index.size();
            
            if(ray_intersect_index.size() > 0)
            {
                for(unsigned int j=0;j<ray_intersect_index.size();j++)
                {
                    roi_img = GetRotateCropImage(text_mat,text_boxes_[ray_intersect_index[j]].box_point_);
                    if(crnn_model_->ModelPredict(roi_img,ocr_str) == DLSUCCESSED)
                    {
                        if(crnn_class_name_.size() != 0)
                        {
                            LOG(ERROR)<<"ocr_str:"<<ocr_str;
                            auto iter = std::find(std::begin(crnn_class_name_), std::end(crnn_class_name_), ocr_str);      //标签有效性检查
                            if(iter != crnn_class_name_.end())                            
                            {   
                                score = text_boxes_[ray_intersect_index[j]].score_;
                                auto iter_split = std::find(std::begin(other_list_), std::end(other_list_), ocr_str);     //检查标签是否是合并标签
                                if(iter_split != other_list_.end())
                                {                                    
                                    GetSplitLabel(ocr_str,text_boxes_[ray_intersect_index[j]],kepointVec,label_str);
                                    break;
                                }
                                temp_text_boxes.emplace_back(text_boxes_[ray_intersect_index[j]]);
                                vec_str.emplace_back(ocr_str);                                
                            } 
                        }
                    }
                }
                if(label_str == UNKNOWN_KNOB && vec_str.size() > 0)
                {
                    GetIntersectLabel(vec_str,temp_text_boxes,label_str); 
                }               
            }
            
            if(label_str == UNKNOWN_KNOB )
            {
                for(unsigned int j=0;j<text_boxes_.size();j++)
                {
                    roi_img = GetRotateCropImage(text_mat,text_boxes_[j].box_point_);                
                    crnn_model_->ModelPredict(roi_img,ocr_str);            
                    if(crnn_class_name_.size() != 0)
                    {
                        LOG(ERROR)<<"ocr_str:"<<ocr_str;
                        auto iter = std::find(std::begin(crnn_class_name_), std::end(crnn_class_name_), ocr_str);
                        if(iter != crnn_class_name_.end())
                        {
                            auto iter_split = std::find(std::begin(other_list_), std::end(other_list_), ocr_str);   //检查标签是否是合并标签
                            if(iter_split != other_list_.end())
                            {
                                GetSplitLabel(ocr_str,text_boxes_[j],kepointVec,label_str);
                                break;
                            }
                            temp_text_boxes.emplace_back(text_boxes_[j]);
                            vec_str.emplace_back(ocr_str);
                        }
                    }                
                }
                if(label_str == UNKNOWN_KNOB && vec_str.size() > 0)
                {
                    GetUnIntersectLabel(vec_str,temp_text_boxes,kepointVec,label_str,score);
                }   
            }           
            result.value = label_str;    
            result.score = score;                          
            output_info.result_list.emplace_back(result);
        }
        else
        {
            LOG(ERROR) << "class idx greater than class name size";
        }
    }
    return DLSUCCESSED;
    
}


int KnobDLAlg::DeInit()
{   
    if(yolov5_model_ != nullptr)
    {
        yolov5_model_->DeInitModel();
    }
    if(keypoint_model_ != nullptr)
    {
        keypoint_model_->DeInitModel();
    }
     if(dbnet_model_ != nullptr)
    {
        dbnet_model_->DeInitModel();
    }
    if(crnn_model_ != nullptr)
    {
        crnn_model_->DeInitModel();
    }
    

    return DLSUCCESSED;
}

REGISTERALG(knob_dlalg, KnobDLAlg);


}