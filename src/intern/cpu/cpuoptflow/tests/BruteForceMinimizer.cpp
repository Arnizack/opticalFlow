#include<gtest/gtest.h>
#include"ImageRGB.h"
#include"FlowField.h"
#include"Minimizer/BruteForceMinimizer.hpp"
#include"FlowFieldVisualizer.h"
#include"Timer.h"
#include"NaiveBilateralFlowFilter.hpp"
#include"ReliabilityMap.h"

class ZeroField : public core::FlowField
{
public:
	using FlowField::FlowField;
	core::FlowVector GetVector(uint32_t x, uint32_t y) const override
	{
		return core::FlowVector(0, 0);
	}
};

TEST(CpuOptFlow, BruteForceMin1)
{
	logger::registerLoggerType(logger::loggerTypes::CONSOLELOGGER,0, "");
	std::string dirpath = __TESTDATADIR__;
	//auto templateFrame = std::make_shared<core::ImageRGB>(dirpath+"/frame10.png");
	//auto nextFrame = std::make_shared<core::ImageRGB>(dirpath + "/frame11.png");
	/*
	auto templateFrame = std::make_shared<core::ImageRGB>(dirpath+"/frame10s.png");
	auto nextFrame = std::make_shared<core::ImageRGB>(dirpath + "/frame11s.png");
	*/
	//auto templateFrame = std::make_shared<core::ImageRGB>(dirpath+"/frame07Small.png");
	//auto nextFrame = std::make_shared<core::ImageRGB>(dirpath + "/frame08Small.png");

	//auto templateFrame = std::make_shared<core::ImageRGB>(dirpath+"/sframe07.jpg");
	//auto nextFrame = std::make_shared<core::ImageRGB>(dirpath + "/sframe08.jpg");

	auto templateFrame = std::make_shared<core::ImageRGB>(dirpath + "/Army07.png");
	auto nextFrame = std::make_shared<core::ImageRGB>(dirpath + "/Army08.png");

	if (nextFrame->GetHeight() == -1 || templateFrame->GetHeight() == -1)
	{
		EXPECT_TRUE(false);
		return;
	}

	std::unique_ptr<core::FlowField> initialFlowField = std::make_unique<ZeroField>(0,0);
	float sigma_d = 5.5;
	float sigma_c = 20;
	uint8_t sampleCount = 64;
	uint8_t searchRegionSize = 10;
	cpu::bilateralfilter::NaiveBilateralFilter filter;
	//107067543.300000 ms
	 //28171712.700000 ms
	cpu::BruteForceMinimizer minimizer(sigma_d, sigma_c, sampleCount, searchRegionSize);

	core::Timer timer;
	auto resultFlow = minimizer.minimize(std::move(initialFlowField), templateFrame, nextFrame);
	
	timer.Stop();
	visualization::FlowFieldVisualizer visualizer;

	auto resultImg = visualizer.visualize(resultFlow, visualization::SCALE::LINEAR);
	resultImg.Save("Tracking.png");

	resultFlow.SaveCsv("tracking.csv");

	auto reliabilityMap = cpu::bilateralfilter::ReliabilityMap(resultFlow, *templateFrame, *nextFrame, 20);

	reliabilityMap.toImage().Save("ReliabilityMap.png");
	/*
	for (uint32_t x = 0; x < templateFrame->GetWidth(); x++)
	{
		for (uint32_t y = 0; y < templateFrame->GetHeight(); y++)
		{
			logger::log(10, "Reliability: %f", reliabilityMap.GetReliability(x, y ));
		}
	}
	*/

	resultFlow = filter.filter(reliabilityMap, resultFlow, *templateFrame, sigma_d, sigma_c);
	
	resultFlow.SaveCsv("trackingFiltered.csv");
	
	resultImg = visualizer.visualize(resultFlow, visualization::SCALE::LINEAR);
	resultImg.Save("TrackingSmoothed.png");

}