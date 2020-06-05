#include <windows.h>
#include <ppl.h>
#include <iostream>
#include <algorithm>
#include <array>

#include<gtest/gtest.h>
#include<KDTree.h>
#include"logger.hpp"
#include<chrono>
#include"KDTreeVisualizer.h"
#include"KDTreeAnalyser.hpp"

#include"Timer.h"
//#include<amp.h>

#include <iostream>


float calcWeight(uint32_t x1, uint32_t y1, unsigned char r1, unsigned char g1, unsigned char b1,
	uint32_t x2, uint32_t y2, unsigned char r2, unsigned char g2, unsigned char b2, float sigma_color, float sigma_distance)
{
	//distance
	float diffX = x2 - x1;
	float diffY = y2 - y1;
	float diffR = r2 - r1;
	float diffG = g2 - g1;
	float diffB = b2 - b1;

	diffX /= sigma_distance;
	diffY /= sigma_distance;
	diffR /= sigma_color;
	diffG /= sigma_color;
	diffB /= sigma_color;

	float distanceSqr = diffX * diffX + diffY * diffY + diffR * diffR + diffG * diffG + diffB * diffB;
	return exp(-distanceSqr);
}

TEST(KDTree, QueryTree)
{
	printf("test\n");
	logger::registerLoggerType(logger::loggerTypes::CONSOLELOGGER, 10,  "logQuery.txt");
	std::string fotoPath = __TESTDATADIR__;
	fotoPath += "\\Lola.png";
	
	std::shared_ptr<core::ImageRGB> imagePtr = std::make_shared<core::ImageRGB>(fotoPath);
	
	
	kdtree::KDTree tree(imagePtr);
	
	float sigma_color = 80;
	float sigma_distance = 20;
	int sampleCount = 32;
	
	logger::log(40, "Build");
	core::Timer timer1(40);
	kdtree::KDTreeData data = tree.Build(sigma_distance, sigma_color, sampleCount);
	timer1.Stop();

	core::ImageRGB resultImg(imagePtr->GetWidth(), imagePtr->GetHeight());
	/*
	kdtree::KDTreeAnalyser analyser;
	analyser.logTreeLevelCount(data, 40);
	analyser.logLeftPath(data, 40);
	*/
	//kdtree::KDTreeVisualizer treeVis(*data);

	//treeVis.write("graph.txt");

	

	core::Timer timer(40);

	concurrency::parallel_for(0, (int)imagePtr->GetWidth(),
		[imagePtr, &resultImg, &data](int x) {
		//for (uint32_t x = 0; x <  imagePtr->GetWidth() ; x++)


		for (uint32_t y = 0; y < imagePtr->GetHeight(); y++)
		{

			auto color = imagePtr->GetPixel(x, y);

			std::vector<kdtree::KDResult> results = kdtree::queryKDTree(data, x, y, color);

			float divider = 0;
			core::Color resultColor;
			float red = 0;
			float green = 0;
			float blue = 0;
			int resultCount = results.size();

			float level = 0;

			for (const auto& result : results)
			{
				//float weight = calcWeight(x,y,color.Red,color.Green,color.Blue,result.X,result.Y,result.R,result.G,result.B,sigma_color,sigma_distance);
				float weight = result.Weight;
				level += result.Level;
				red += result.R * weight;
				green += result.G * weight;
				blue += result.B * weight;
				divider += weight;

			}

			level /= results.size();

			//logger::log(20, "Result Count: %d", resultCount);
			//logger::log(20, "Avg Level Count %f", level);

			red /= divider;
			green /= divider;
			blue /= divider;
			resultColor.Red = static_cast<unsigned char> (red);
			resultColor.Green = static_cast<unsigned char> (green);
			resultColor.Blue = static_cast<unsigned char> (blue);
			resultImg.SetPixel(x, y, resultColor);


		}

		//logger::log(40, "x: %d", x);
	}
	);
	timer.Stop();
	resultImg.Save("BilateralBlureLola.png");
	
}