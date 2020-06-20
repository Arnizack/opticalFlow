#pragma once
#include<stdint.h>
#include<vector>
#include<string>
namespace utilities
{
	struct FlowField
	{
		uint32_t width;
		uint32_t height;
		std::vector<float> data;
	};

	bool loadFlowFromCSV(std::string path, FlowField& field);
	
	bool saveFlowToCSV(std::string path, FlowField& field, char seperator = ',');

}