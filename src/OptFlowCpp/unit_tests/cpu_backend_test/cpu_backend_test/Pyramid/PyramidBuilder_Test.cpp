#include "cpu_backend/pyramid/PyramidBuilder.h"
#include "cpu_backend/ArrayFactory.h"
#include "cpu_backend/Scaler.h"
#include "cpu_backend/problem/ProblemFactory.h"

#include "gtest/gtest.h"

namespace cpu_backend
{
	namespace testing
	{
		//TEST(PyramidBuilder, IGrayPenaltyCrossProblem)
		//{
		//	using PtrIGrayPenaltyCrossProblem = std::shared_ptr<core::IGrayPenaltyCrossProblem>;

		//	std::shared_ptr<ProblemFactory> ptr_problem_factory = std::make_shared<ProblemFactory>(ProblemFactory(nullptr));

		//	std::shared_ptr<ArrayFactory<float, 2>> ptr_arr_factory_2D = std::make_shared<ArrayFactory<float, 2>>(ArrayFactory<float, 2>());
		//	std::shared_ptr<ArrayFactory<float, 3>> ptr_arr_factory_3D = std::make_shared<ArrayFactory<float, 3>>(ArrayFactory<float, 3>());
		//	
		//	//needs IGrayPenaltyCrossProblem Scaler
		//	std::shared_ptr<Scaler<float, 2>> ptr_scaler_2D = std::make_shared<Scaler<float, 2>>(Scaler<float, 2>(ptr_arr_factory_2D));
		//	std::shared_ptr<Scaler<float, 3>> ptr_scaler_3D = std::make_shared<Scaler<float, 3>>(Scaler<float, 3>(ptr_arr_factory_3D));

		//	PyramidBuilder<PtrIGrayPenaltyCrossProblem> pyramid_builder(ptr_problem_factory, ptr_scaler_2D);

		//	/*
		//	* SET RESOLUTIONS
		//	*/

		//	std::vector<std::array<size_t, 2>> resolutions = { {2,2}, {4,4}, {6,6} };
		//	pyramid_builder.SetResolutions(resolutions);

		//	auto last_level = ptr_problem_factory->CreateGrayPenaltyCrossProblem();
		//	last_level->FirstFrame = ptr_arr_factory_2D->Full(1, { 6,6 });
		//	last_level->SecondFrame = ptr_arr_factory_2D->Full(2, { 6,6 });
		//	last_level->CrossFilterImage = ptr_arr_factory_3D->Full(3, {6,6,3});
		//	last_level->PenaltyFunc;

		//	auto pyramid = pyramid_builder.Create(last_level);

		//	/*for (int i = 0; i < resolutions.size() + 1; i++)
		//	{
		//		if (i == resolutions.size())
		//		{
		//			EXPECT_EQ(pyramid->IsEndLevel(), true);
		//			return;
		//		}
		//		else
		//			EXPECT_EQ(pyramid->IsEndLevel(), false);

		//		auto out_temp = pyramid->NextLevel();

		//		EXPECT_EQ(out_temp->FirstFrame->Shape[0], resolutions[i+1][0]);
		//		EXPECT_EQ(out_temp->FirstFrame->Shape[1], resolutions[i+1][1]);
		//	}*/

		//	int i = 1;

		//	while (!pyramid->IsEndLevel())
		//	{
		//		auto out_temp = pyramid->NextLevel();

		//		EXPECT_EQ(out_temp->FirstFrame->Shape[0], resolutions[i][0]);
		//		EXPECT_EQ(out_temp->FirstFrame->Shape[1], resolutions[i][1]);

		//		i++;
		//	}

		//	EXPECT_EQ(pyramid->IsEndLevel(), true);

		//	/*
		//	* SET Scale Factors
		//	*/

		//	std::vector<double> factors = { 1.0/3.0, 2.0/3.0, 1 };
		//	pyramid_builder.SetScaleFactors(factors);

		//	pyramid = pyramid_builder.Create(last_level);

		//	i = 1;

		//	while (!pyramid->IsEndLevel())
		//	{
		//		auto out_temp = pyramid->NextLevel();

		//		EXPECT_EQ(out_temp->FirstFrame->Shape[0], resolutions[i][0]);
		//		EXPECT_EQ(out_temp->FirstFrame->Shape[1], resolutions[i][1]);

		//		i++;
		//	}

		//	EXPECT_EQ(pyramid->IsEndLevel(), true);

		//	/*
		//	* SET Scale Factor
		//	*/

		//	std::array<size_t, 2> min_res = { {2,2} };
		//	const double factor = 2;
		//	pyramid_builder.SetScaleFactor(factor, min_res);

		//	pyramid = pyramid_builder.Create(last_level);

		//	for (int i = 0; i < 3; i++)
		//	{
		//		if (i == 1)
		//		{
		//			EXPECT_EQ(pyramid->IsEndLevel(), false);
		//			auto out_temp = pyramid->NextLevel();

		//			EXPECT_EQ(out_temp->FirstFrame->Shape[0], 6);
		//			EXPECT_EQ(out_temp->FirstFrame->Shape[1], 6);
		//		}
		//		else if(i == 2)
		//			EXPECT_EQ(pyramid->IsEndLevel(), true);
		//		else
		//		{
		//			EXPECT_EQ(pyramid->IsEndLevel(), false);

		//			auto out_temp = pyramid->NextLevel();

		//			min_res[0] *= factor;
		//			min_res[1] *= factor;

		//			EXPECT_EQ(out_temp->FirstFrame->Shape[0], min_res[0]);
		//			EXPECT_EQ(out_temp->FirstFrame->Shape[1], min_res[1]);
		//		}
		//	}
		//}
	}
}