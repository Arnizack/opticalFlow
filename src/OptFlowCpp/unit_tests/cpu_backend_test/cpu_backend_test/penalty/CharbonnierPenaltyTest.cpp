#include<gtest/gtest.h>
#include"cpu_backend/penalty/CharbonnierPenalty.h"

namespace cpu_backend
{
	namespace testing
	{
		TEST(CharbonnierPenaltyTest, test1)
		{
			std::vector<double> blend_factors = { 0,0.5,1 };
			std::vector<double> x_list = { 0,0.2,0.6,1,5 };
			std::vector<double> expected_value =
			{
				0.0, 0.04000000000000001, 0.36,
				1.0, 25.0, 0.0009976311574844397,
				0.13746321574602804, 0.4957233283980433, 1.0000002249999382,
				14.628349844612258, 0.0019952623149688794, 0.2349264314920561,
				0.6314466567960866, 1.0000004499998763, 4.256699689224515
			};
			std::vector<double> expected_1deriv =
			{
				0.0, 0.4, 1.2,
				2.0, 10.0, 0.0,
				0.7285712565757119, 1.073583677086851, 1.449999752500192,
				5.383102956706088, 0.0, 1.0571425131514238,
				0.9471673541737019, 0.8999995050003837, 0.7662059134121761
			};
			std::vector<double> expected_2deriv =
			{
				2.0, 2.0, 2.0,
				2.0, 2.0, 898.8680417359964,
				3.6428562828785593, 1.789306128478085, 1.449999752500192,
				1.0766205913412177, 1795.7360834719927, 5.285712565757119,
				1.5786122569561698, 0.8999995050003837, 0.15324118268243522
			};

			std::vector<double> actual_value;
			std::vector<double> actual_1deriv;
			std::vector<double> actual_2deriv;

			

			auto settings = std::make_shared<CharbonnierPenaltySettings>();
			settings->DefaultBlendFactor = 0;
			settings->Epsilon = 0.001;
			settings->Exponent = 0.45;

			CharbonnierPenalty penalty(settings);

			for(double blend_factor : blend_factors)
			{
				penalty.SetBlendFactor(blend_factor);
				for (double x : x_list)
				{
					actual_value.push_back(penalty.ValueAt(x));
					actual_1deriv.push_back(penalty.FirstDerivativeAt(x));
					actual_2deriv.push_back(penalty.SecondDerivativeAt(x));
				}
			}

			size_t range = expected_value.size();
			for (int i = 0; i < range; i++)
			{
				double near = 0.0001;
				EXPECT_NEAR(expected_value[i], actual_value[i], near);
				EXPECT_NEAR(expected_1deriv[i], actual_1deriv[i], near);
				EXPECT_NEAR(expected_2deriv[i], actual_2deriv[i], near);
			}




		}
	}
}