#include"gtest/gtest.h"
#include<array>
#include"cpu_backend/Array.h"
#include"flow_helper/FlowHelper.h"
#include<filesystem>

namespace flowhelper
{
	TEST(flow_helper, test1)
	{
		std::array<double, 32> expected_flow =
		{0, 1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 };

		std::string path = "test.flo";
		std::array<const size_t, 3> shape = {2, 4,4 };
		auto flow = std::make_shared< cpu_backend::Array<double, 3>>
			(shape, expected_flow.data());

		flowhelper::SaveFlow(path, flow);
		FlowField actual_flow = OpenFlow(path);

		double* actual_data = actual_flow.data->data();
		double* expected_data = expected_flow.data();

		for (int i = 0; i < 16; i++)
		{
			EXPECT_NEAR(actual_data[i], expected_data[i], 0.05);
		}
	}

	TEST(flow_helper, color)
	{
		std::array<double, 32> expected_flow =
		{ 0, 1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 };

		std::string path = "testflow.png";
		std::array<const size_t, 3> shape = { 2, 4,4 };
		auto flow = std::make_shared< cpu_backend::Array<double, 3>>
			(shape, expected_flow.data());

		
		SaveFlow2Color(path, flow);
		
	}
	#define WINDOWS
	#ifdef WINDOWS
	#include <direct.h>
	#define GetCurrentDir _getcwd
	#else
	#include <unistd.h>
	#define GetCurrentDir getcwd
	#endif

	std::string get_current_dir() {
		char buff[FILENAME_MAX]; //create string buffer to hold path
		GetCurrentDir(buff, FILENAME_MAX);
		std::string current_working_dir(buff);
		return current_working_dir;
	}
	TEST(flow_helper, color2)
	{

		std::string path = "DimetrodonConvertet.png";

		auto cwd = get_current_dir();

		FlowField flow = OpenFlow("..\\..\\..\\..\\..\\..\\resources\\eval-twoframes-groundtruth\\Dimetrodon\\flow10.flo");

		

		SaveFlow2Color(path, flow);

	}
}