#ifndef DL_FORMAT_H
#define DL_FORMAT_H

#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;


	namespace Meter_jsons {

	

	struct MeterConfigInfo{		    
		std::vector<std::string> recognition_list;		
	};

}


int GetMeterConfigFromJson(const std::string josn_file,Meter_jsons::MeterConfigInfo &meterCfg);




#endif