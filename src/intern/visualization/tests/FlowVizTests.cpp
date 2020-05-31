#include<gtest/gtest.h>
#include"FlowFieldVisualizer.h"

TEST(visualization, FlowViz2)
{
	visualization::FlowFieldVisualizer visualizer;
	int width = 1000;
	int heigth = 1000;
	core::FlowField flow(width, heigth);
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < heigth; y++)
		{
			int xval = sin(((float)x*y) / 10000)*1000;
			int yval = cos(((float)y*y) / 10000)*1000;
			core::FlowVector test;
			flow.SetVector(x, y, core::FlowVector(xval, yval));
		}
	}

	auto img = visualizer.visualize(flow,visualization::SCALE::LOG);
	img.Save("testFlowFliedViz.png");
}

TEST(visualization, FlowViz3)
{
	visualization::FlowFieldVisualizer visualizer;
	int width = 1000;
	int heigth = 1000;
	core::FlowField flow(width, heigth);
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < heigth; y++)
		{
			int xval = sin(((float)x) * 3.14 / width) * 10;
			int yval = cos(((float)y)*3.14 / heigth) * 10;
			core::FlowVector test;
			flow.SetVector(x, y, core::FlowVector(xval, yval));
		}
	}

	auto img = visualizer.visualize(flow, visualization::SCALE::LOG);
	img.Save("testFlowFliedViz2.png");
}