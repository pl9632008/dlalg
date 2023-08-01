#include "face_recognition.h"
#include "dl_algorithm.h"
#include "dl_common.h"
#include <string>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>


namespace yjh_deeplearning {


class FaceRecognitionAlgorithm::FaceRecognitionAlgorithmImp {
	public:
		FaceRecognitionAlgorithmImp()=default;
		~FaceRecognitionAlgorithmImp()=default;
		int Init(const std::string &alg_conf_file, int gpu_index);
        int ProcessImage(const std::shared_ptr<void> &ori_img,std::vector<int> &face_det,std::vector<float> &features);
		bool IsSamePeople(std::vector<float> &face_1,std::vector<float> &face_2,float threshold);
      
	private:
		AIAlgorithm dlalg_;
				
	};


int FaceRecognitionAlgorithm::FaceRecognitionAlgorithmImp::Init(const std::string &alg_conf_file, int gpu_index)
{
	return dlalg_.Init(alg_conf_file,gpu_index);
}


int FaceRecognitionAlgorithm::FaceRecognitionAlgorithmImp::ProcessImage(const std::shared_ptr<void> &ori_img,std::vector<int> &face_det,std::vector<float> &features)
{	
    AIInputInfo input_info;
    AIOutputInfo output_info;
  
    input_info.src_mat.emplace_back(ori_img);  
    int ret = dlalg_.Inference(input_info,output_info); 
    if(ret != 0 )
    {
        return ret;
    }   
    face_det.resize(4);
    face_det[0] = output_info.result_list[0].center_x;
    face_det[1] = output_info.result_list[0].center_y;
    face_det[2] = output_info.result_list[0].width;
    face_det[3] = output_info.result_list[0].height;
    std::shared_ptr<cv::Mat> result = std::static_pointer_cast<cv::Mat>(output_info.result_list[0].dst_mat);
    features.resize(result->size[0]*result->size[1]);
    std::memcpy(features.data(), result->data,features.size()*sizeof(float ));//[3]
	return DLSUCCESSED;
}

bool FaceRecognitionAlgorithm::FaceRecognitionAlgorithmImp::IsSamePeople(std::vector<float> &face_1,std::vector<float> &face_2,float threshold)
{   
    cv::Mat first,second;
    // LOG(ERROR) << "face_1 size: " << face_1.size()<<" "<<"face_2 size: " << face_2.size();
    try{
        first = cv::Mat(1,face_1.size(),CV_32FC1,cv::Scalar(0.0));
        second = cv::Mat(1,face_2.size(),CV_32FC1,cv::Scalar(0.0));
        std::memcpy(first.data, face_1.data(), face_1.size()* sizeof(float));
        std::memcpy(second.data, face_2.data(), face_2.size()* sizeof(float));
        double dotSum=first.dot(second);//内积
        double normFirst=cv::norm(first);//取模
        double normSecond=cv::norm(second); 
        if(normFirst!=0 && normSecond!=0){
            // LOG(ERROR)<<"simility: "<<dotSum/(normFirst*normSecond);
            if(dotSum/(normFirst*normSecond)>threshold)
            {
                return true;
            }
        }
    }
    catch (cv::Exception &e)
    {
        // output exception information
        LOG(ERROR) << "message: " << e.what();
        return false;
    }
    return false;
}


int FaceRecognitionAlgorithm::Init(std::string alg_conf_file,int gpu_index)
{
	face_alg_imp_ = std::make_shared<FaceRecognitionAlgorithmImp>();
	return face_alg_imp_->Init(alg_conf_file,gpu_index);	
}



int FaceRecognitionAlgorithm::ProcessImage(const std::shared_ptr<void> &ori_img,std::vector<float> &features)
{
    std::vector<int> face_det;
	return 	face_alg_imp_->ProcessImage(ori_img,face_det,features);
}

int FaceRecognitionAlgorithm::ProcessImage(const std::shared_ptr<void> &ori_img,std::vector<int> &face_det,std::vector<float> &features)
{
    return 	face_alg_imp_->ProcessImage(ori_img,face_det,features);
}


bool FaceRecognitionAlgorithm::IsSamePeople(std::vector<float> &face_1,std::vector<float> &face_2,float threshold)
{
	return 	face_alg_imp_->IsSamePeople(face_1, face_2, threshold);
}


}

yjh_deeplearning::FaceRecognitionAlgorithm *GetFaceRecognitionAlgorithmInstance()
{
    if(!google::IsGoogleLoggingInitialized())
	{
		FLAGS_log_dir = "./";
		FLAGS_logbufsecs = 0;
		FLAGS_max_log_size = 2;
		google::InitGoogleLogging("YJHDLALG");
	}
    //  LOG(ERROR) << "Get face recognition instance! ";
    return new yjh_deeplearning::FaceRecognitionAlgorithm();
}

void DestoryFaceRecognitionAlgorithmInstance(yjh_deeplearning::FaceRecognitionAlgorithm *facepointer)
{
    if(facepointer != nullptr)
    {
        delete facepointer;
        facepointer = nullptr;
    }
}

int InitFaceRecognitionAlgorithm(yjh_deeplearning::FaceRecognitionAlgorithm *self,char *alg_conf_file,int gpu_index)
{
   
    // LOG(ERROR) << "init face recognition start "<<alg_conf_file<<" "<<gpu_index;
    if(self != nullptr)
    {
        return self->Init(alg_conf_file,gpu_index);
    }
    else
    {
        return yjh_deeplearning::DLFAILED;
    }
}

int ProcessFaceImage(yjh_deeplearning::FaceRecognitionAlgorithm *self,char *img_ptr,int img_length,float *features,int *size)
{
    // LOG(ERROR) << "ProcessFaceImage start "<<self<<" "<< (void*)img_ptr << " "<<img_length <<" "<<*size<<" "<<(int)img_ptr[0]<<" "<<(int)img_ptr[1];
    if(self != nullptr && features != nullptr && img_ptr != nullptr)
    {
        std::vector<float> vecFeature;
        std::vector<uchar> data(img_length);      
        std::memcpy(data.data(),img_ptr,img_length);
        cv::Mat ori_img;
        try{
            ori_img = cv::imdecode(data, cv::IMREAD_COLOR);
        }
        catch (cv::Exception &e)
        {
            // output exception information
            LOG(ERROR) << "message: " << e.what();
            return yjh_deeplearning::YJH_AI_FACE_IMAGE_ERROR;
        }      
        if(ori_img.data == nullptr || ori_img.empty() || ori_img.total() == 0)
        {
            LOG(ERROR) << "input img is error! ori_img.data is "<<ori_img.data;
            LOG(ERROR) << "ori_img empty() is "<<ori_img.empty();
            LOG(ERROR) << "ori_img total() is "<<ori_img.total();       
            return yjh_deeplearning::YJH_AI_FACE_IMAGE_ERROR;
        }
        
        // cv::imwrite("ori_img.jpg",ori_img);
        
        int ret = self->ProcessImage(std::make_shared<cv::Mat>(ori_img),vecFeature);
        if(ret != yjh_deeplearning::DLSUCCESSED)
        {
            return ret;
        }
        if(*size < vecFeature.size())
        {
            LOG(ERROR) << "features space is not enough, need "<<vecFeature.size()<<" bytes";
            return yjh_deeplearning::DLFAILED;
        }
        *size = vecFeature.size(); 
        std::memcpy(features, vecFeature.data(), vecFeature.size()* sizeof(float)); 
        return yjh_deeplearning::DLSUCCESSED;
    }
    else
    {
        LOG(ERROR) << "ProcessFaceImage null ptr ";
        return yjh_deeplearning::DLFAILED;
    }
}


int ProcessFaceLocalImage(yjh_deeplearning::FaceRecognitionAlgorithm *self,char *img_ptr,int *center_x,int *center_y,int *width,int *height,float *features,int *size)
{
    if(self != nullptr && features != nullptr && img_ptr != nullptr)
    {
        std::vector<float> vecFeature;       
        cv::Mat ori_img;
        try{
            ori_img = cv::imread(img_ptr, cv::IMREAD_COLOR);
        }
        catch (cv::Exception &e)
        {
            // output exception information
            LOG(ERROR) << "message: " << e.what();
            return yjh_deeplearning::YJH_AI_FACE_IMAGE_ERROR;
        }      
        if(ori_img.data == nullptr || ori_img.empty() || ori_img.total() == 0)
        {
            LOG(ERROR) << "input img is error! ori_img.data is "<<ori_img.data;
            LOG(ERROR) << "ori_img empty() is "<<ori_img.empty();
            LOG(ERROR) << "ori_img total() is "<<ori_img.total();       
            return yjh_deeplearning::YJH_AI_FACE_IMAGE_ERROR;
        }
        
        // cv::imwrite("ori_img.jpg",ori_img);
        std::vector<int> det_face;
        int ret = self->ProcessImage(std::make_shared<cv::Mat>(ori_img),det_face,vecFeature);
        if(ret != yjh_deeplearning::DLSUCCESSED)
        {
            return ret;
        }
        if(*size < vecFeature.size() || det_face.size() != 4 )
        {
            LOG(ERROR) << "features space is not enough, need "<<vecFeature.size()<<" bytes" << "or det face size is error";
            return yjh_deeplearning::DLFAILED;
        }
        *center_x = det_face[0];
        *center_y = det_face[1];
        *width = det_face[2];
        *height = det_face[3];
       
        *size = vecFeature.size(); 
        std::memcpy(features, vecFeature.data(), vecFeature.size()* sizeof(float)); 
        return yjh_deeplearning::DLSUCCESSED;
    }
    else
    {
        LOG(ERROR) << "ProcessFaceImage null ptr ";
        return yjh_deeplearning::DLFAILED;
    }
}




bool IsSamePeople(yjh_deeplearning::FaceRecognitionAlgorithm *self,float *face_1,int size_1,float *face_2,int size_2,float threshold)
{   
    // LOG(ERROR) << "IsSamePeople start "<<self<<" "<< face_1 << " "<<face_2 <<" "<< size_1 <<" "<<size_2<<" "<<threshold;

    if(self != nullptr && face_1 != nullptr && face_2 != nullptr && size_1 == size_2)
    {   
        std::vector<float> feat1,feat2;
        feat1.resize(size_1);
        std::memcpy(feat1.data(), face_1,size_1*sizeof(float ));//[3]
        feat2.resize(size_2);
        std::memcpy(feat2.data(), face_2,size_2*sizeof(float ));//[3]        
        return self->IsSamePeople(feat1,feat2,threshold);
    }
    else
    {
        LOG(ERROR) << "IsSamePeople null ptr "<<self<<" "<< face_1 << " "<<face_2 <<" "<< size_1 <<" "<<size_2<<" "<<threshold;
        return false;
    }
}