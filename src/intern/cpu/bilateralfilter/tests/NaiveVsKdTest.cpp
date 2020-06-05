#include<gtest/gtest.h>
#include"FlowField.h"
#include"ReliabilityFactory.hpp"
#include"FlowFieldVisualizer.h"
#include"BilateralFlowFilter.h"
#include"KDTree.h"
#include"NaiveBilateralFlowFilter.hpp"
#include"Timer.h"
#include"logger.hpp"

using namespace cpu::bilateralfilter;

TEST(bilateralfilter, naiveVskd)
{
	logger::registerLoggerType(logger::loggerTypes::CONSOLELOGGER, 10, "");
	std::string dirpath = __TESTDATADIR__;
	auto templatePath = dirpath + "/frame08.png";
	auto searchPath = dirpath + "/frame09.png";

	visualization::FlowFieldVisualizer visualizer;
	

	float delta_c = 80;
	float delta_d = 20;

	std::shared_ptr<core::ImageRGB> templateImg = std::make_shared<core::ImageRGB>(templatePath);
	std::shared_ptr<core::ImageRGB> searchImg = std::make_shared<core::ImageRGB>(searchPath);
	int width = templateImg->GetWidth();
	int heigth = templateImg->GetHeight();

	core::FlowField flow(width, heigth);
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < heigth; y++)
		{
			int xval = sin(((float)x) * 3.14 / width) * 10;
			int yval = cos(((float)y) * 3.14 / heigth) * 10;
			core::FlowVector test;
			flow.SetVector(x, y, core::FlowVector(xval, yval));
		}
	}

	auto preFiltered = visualizer.visualize(flow);
	preFiltered.Save("preFiltered.png");

	kdtree::KDTree tree(templateImg);
	auto data = tree.Build(delta_d, delta_d, 32);
	ReliabilityMap reliableMap(flow,*templateImg,*searchImg,20);
	
	logger::log(20, "KDTree:");
	core::Timer timerKD(20);

	BilateralFlowFilter filterKd;
	auto kdFlowFiltered = filterKd.filter(flow, reliableMap, data, *templateImg);
	
	timerKD.Stop();

	logger::log(20, "Naive:");
	core::Timer timerNaive(20);

	NaiveBilateralFilter naiveFilter;
	auto naiveFlowFiltered = naiveFilter.filter(reliableMap, flow, *templateImg, delta_d, delta_c);
	timerNaive.Stop();
	
	auto kdImg = visualizer.visualize(kdFlowFiltered);
	kdImg.Save("KDFlowFitered.png");
	
	auto naiveImg = visualizer.visualize(naiveFlowFiltered);
	naiveImg.Save("NaiveFlowFitered.png");
	
	core::FlowField flowDiff(width, heigth);
	float diffAverageX = 0;
	float diffAverageY = 0;
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < heigth; y++)
		{

			auto vec1 = naiveFlowFiltered.GetVector(x, y);
			auto vec2 = kdFlowFiltered.GetVector(x, y);
			auto diff = vec1 - vec2;
			flowDiff.SetVector(x, y, diff);
			diffAverageX += diff.vector_X* diff.vector_X;
			diffAverageY += diff.vector_Y* diff.vector_Y;
		}
	}
	diffAverageX /= width * heigth;
	diffAverageY /= width * heigth;
	
	logger::log(20, "Log X %f", diffAverageX);

	logger::log(20, "Log Y %f", diffAverageY);

	auto diffImg = visualizer.visualize(flowDiff);
 	diffImg.Save("Diff.png");

}