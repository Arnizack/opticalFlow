#pragma once
#include"core/IArray.h"
#include<memory>
#include<vector>
#include<string>

namespace flowhelper
{
	struct FlowField
	{
		size_t width;
		size_t height;
		std::shared_ptr<std::vector<double>> data;

		double& GetXFlow(size_t x, size_t y);

		double& GetYFlow(size_t x, size_t y);
	};

	FlowField OpenFlow(std::string filepath);
	void SaveFlow(std::string filepath, FlowField flow);
	void SaveFlow(std::string filepath, std::shared_ptr<core::IArray<double, 3>> flow);
	
	void SaveFlow2Color(std::string filepath, FlowField flow);
	void SaveFlow2Color(std::string filepath, std::shared_ptr<core::IArray<double, 3>> flow);

}