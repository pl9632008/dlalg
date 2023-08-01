#ifndef DL_FORMAT_H
#define DL_FORMAT_H

#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace yjh_deeplearning{
	
	namespace dlalg_jsons {

	struct PreprocessInfo
	{
		int img_width{0};
		int img_height{0};
		int img_channel;
		float img_max_value{255.0};
		int short_size;
    	int max_size;    
		int padding{0};
		std::vector<float> img_mean;
		std::vector<float> img_std;
		bool is_rgb;
		float img_scale;
	};

	struct ModelInfo{		
		std::string model_name;
		int gpu_index;		
		unsigned int batch_size;
		std::vector<std::string> auto_preprocess{};
		std::vector<PreprocessInfo> preprocess_list{};	
		std::string weight_path;
		std::string onnxruntime_customop_library;
		int onnxruntime_intraOpNum;

		//opencv 框架推理时，具体模型的类型，典型如ONNX,Caffe,Darknet,Torch 等
		std::string model_type;
		std::string infer_engine;
		bool is_half;

		bool is_dynamic_infer{false};

		std::string cfg_path;
		std::vector<std::string>  class_name{};

		std::vector<std::string> name_input{};
		std::vector<std::string> name_output{};
		std::vector<std::vector<float>> anchor{};
		std::vector<float> stride{};
		std::map<std::string,float> class_thresh{};		
		float conf_threshold;
		float iou_threshold;
		float obj_threshold;
		float mask_threshold;	

		std::vector<std::string>  other_list{};

		std::map<std::string,float> other_conf_thresh{};  //其他配置阈值，有的模型阈值较多，用户自定义用map存储

		std::map<std::string,std::string> other_conf;  //其他配置，用户自定义
		
		bool is_draw;	
		bool multi_label;
	};	

	struct AlgInfo{
		std::string alg_name;        
		std::vector<ModelInfo> model_list;
		
	};

	void from_json(const json& j, ModelInfo& mi);	
	void from_json(const json& j, PreprocessInfo& pi);

}


int GetConfigFromJson(const std::string josn_file,dlalg_jsons::AlgInfo &algCfg);


}

#endif