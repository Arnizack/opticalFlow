#include"cpu_backend/flow/CrossBilateralMedianFilter.h"
#include"gtest/gtest.h"
#include<sstream>
#include<fstream>

namespace cpu_backend
{
	
	TEST(CrossBilateralMedianFilterTest, test1)
	{
		
		double flow[70] =
		{
			-4.45333189e-02, -2.64400687e-02, -1.74602378e-02, -1.21782225e-02, -8.27281123e-03, -2.08896372e-03,  3.47853211e-04,
			-4.58008989e-02, -4.60611808e-02, -3.59827879e-02, -1.66246419e-02, -1.57139908e-02,  2.13075254e-03,  4.14972210e-03,
			-1.98916287e-02, -3.11841566e-02, -2.54235660e-02, -1.37003479e-02, 5.22110803e-03,  1.40253807e-02,  9.04690305e-03,
			-1.93571653e-02, -2.75315915e-02, -3.04310572e-02, -3.87269319e-03, -2.40842453e-03, -9.79753342e-05,  1.24949200e-03,
			-2.39405666e-02, -3.18769384e-02, -3.58664626e-02, -2.62541840e-02,-2.15233984e-02, -1.73480988e-02, -7.34160361e-03,

			-8.94368966e-03, -2.52710902e-03,  9.82591755e-03, -2.06665272e-03, -2.18519877e-02, -3.47381975e-02, -2.32669439e-02,
			7.89709688e-03,  1.27247236e-02,  9.88429563e-03,  2.58216413e-03, -1.71296892e-02, -4.36177870e-02, -3.39598290e-02,
			7.46745338e-03,  1.47848070e-02,  1.80651884e-02,  1.60903127e-02, 1.12952349e-02, -4.37088238e-02, -3.63345471e-02,
			4.27294513e-03,  1.80658929e-02,  3.68889628e-02,  3.22158528e-02, 2.32580297e-02, -4.84490709e-02, -3.59219861e-02,
			4.52615972e-03,  1.87729092e-02,  3.77690217e-02,  4.15485472e-02, 1.59657358e-02, -3.55334945e-02, -2.93318388e-02

		};
		float image[105] = { 0 };

		double log_occlusion[35] =
		{
			-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -6.42494207e-05, -1.00711551e-03, -0.00000000e+00, -0.00000000e+00,
			-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -2.91971177e-04, -4.51870439e-04, -0.00000000e+00, -0.00000000e+00,
			-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -5.61807539e-03, -6.46456074e-03, -0.00000000e+00,
			-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -4.92821145e-04, -2.44278579e-02, -1.72047294e-02, -0.00000000e+00,
			-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -3.47589890e-03, -1.79334304e-02, -6.26310905e-03, -0.00000000e+00

		};

		double expected_flow[70] =
		{
			 - 0.01746024, - 0.01746024, - 0.01746024, - 0.01746024, - 0.01746024, - 0.01746024,
			 - 0.0173481,  - 0.01746024, - 0.01746024, - 0.01746024,  - 0.0173481,  - 0.0173481,
			 - 0.0173481,   - 0.0173481,  - 0.0173481,  - 0.0173481,  - 0.0173481,  - 0.0173481,
			 - 0.0173481,   - 0.0173481,  - 0.0173481,  - 0.0173481,  - 0.0173481,  - 0.0173481,
			 - 0.0173481,   - 0.0173481, - 0.01662464, - 0.01662464,  - 0.0173481, - 0.01662464,
			 - 0.01662464, - 0.01662464, - 0.01662464, - 0.01662464, - 0.01662464,   0.00427295,
			   0.00427295,   0.00427295,   0.00427295,   0.00427295,   0.00427295,   0.00452616,
			   0.00427295,   0.00427295,   0.00427295,   0.00452616,   0.00452616,   0.00452616,
			   0.00452616,   0.00452616,   0.00452616,   0.00452616,   0.00452616,   0.00452616,
			   0.00452616,   0.00452616,   0.00452616,   0.00452616,   0.00452616,   0.00452616,
			   0.00452616,   0.00746745,   0.00746745,   0.00452616,   0.00746745,   0.00746745,
			   0.00746745,   0.00746745,   0.00746745,   0.00746745

		};
		
		std::shared_ptr<DerivativeCalculator<double>> flow_deriv_calc
			= std::make_shared<DerivativeCalculator<double>>();
		double sigma_div = 0.3;
		double sigma_error = 20;
		double filter_influence = 5;
		double auxilary_influence = 3.921568627450981e-07;
		double sigma_distance = 7;
		double sigma_color = 0.027450980392156862;
		int filter_length = 15;
		
		std::array<const size_t, 3> img_shape = { 3,5,7 };
		auto ptr_image = std::make_shared<Array<float, 3>>(img_shape, image);

		std::array<const size_t, 3> flow_shape = { 2,5,7 };
		auto ptr_flow = std::make_shared<Array<double, 3>>(flow_shape, flow);
		auto ptr_flow_result = std::make_shared<Array<double, 3>>(flow_shape, flow);

		std::array<const size_t, 2> occ_shape = { 5,7 };
		auto ptr_log_occlusion = std::make_shared<Array<double, 2>>(occ_shape, log_occlusion);

		
		auto settings = std::make_shared<CrossMedianFilterSettings>();
		settings->SigmaDiv = sigma_div;
		settings->SigmaError = sigma_error;
		settings->FilterInfluence = filter_influence;
		settings->SigmaDistance = sigma_distance;
		settings->SigmaColor = sigma_color;
		settings->FilterLength = filter_length;
		settings->Speedup = false;

		CrossBilateralMedianFilter filter(flow_deriv_calc, settings);
		filter.SetAuxiliaryInfluence(auxilary_influence);
		filter.SetCrossFilterImage(ptr_image);
		filter.ApplyTo(ptr_flow_result, ptr_flow);
		
		for (int i = 0; i < 70; i++)
		{
			EXPECT_NEAR(expected_flow[i], ptr_flow_result->Data()[i],0.0001);
			if (abs(expected_flow[i] - ptr_flow_result->Data()[i]) > 0.0001)
			{
				printf("Test\n");
			}
		}
	}

	TEST(CrossBilateralMedianFilterTest, speedup)
	{

		double flow[70] =
		{
			-4.45333189e-02, -2.64400687e-02, -1.74602378e-02, -1.21782225e-02, -8.27281123e-03, -2.08896372e-03,  3.47853211e-04,
			-4.58008989e-02, -4.60611808e-02, -3.59827879e-02, -1.66246419e-02, -1.57139908e-02,  2.13075254e-03,  4.14972210e-03,
			-1.98916287e-02, -3.11841566e-02, -2.54235660e-02, -1.37003479e-02, 5.22110803e-03,  1.40253807e-02,  9.04690305e-03,
			-1.93571653e-02, -2.75315915e-02, -3.04310572e-02, -3.87269319e-03, -2.40842453e-03, -9.79753342e-05,  1.24949200e-03,
			-2.39405666e-02, -3.18769384e-02, -3.58664626e-02, -2.62541840e-02,-2.15233984e-02, -1.73480988e-02, -7.34160361e-03,

			-8.94368966e-03, -2.52710902e-03,  9.82591755e-03, -2.06665272e-03, -2.18519877e-02, -3.47381975e-02, -2.32669439e-02,
			7.89709688e-03,  1.27247236e-02,  9.88429563e-03,  2.58216413e-03, -1.71296892e-02, -4.36177870e-02, -3.39598290e-02,
			7.46745338e-03,  1.47848070e-02,  1.80651884e-02,  1.60903127e-02, 1.12952349e-02, -4.37088238e-02, -3.63345471e-02,
			4.27294513e-03,  1.80658929e-02,  3.68889628e-02,  3.22158528e-02, 2.32580297e-02, -4.84490709e-02, -3.59219861e-02,
			4.52615972e-03,  1.87729092e-02,  3.77690217e-02,  4.15485472e-02, 1.59657358e-02, -3.55334945e-02, -2.93318388e-02

		};
		float image[105] = { 0 };

		double log_occlusion[35] =
		{
			-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -6.42494207e-05, -1.00711551e-03, -0.00000000e+00, -0.00000000e+00,
			-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -2.91971177e-04, -4.51870439e-04, -0.00000000e+00, -0.00000000e+00,
			-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -5.61807539e-03, -6.46456074e-03, -0.00000000e+00,
			-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -4.92821145e-04, -2.44278579e-02, -1.72047294e-02, -0.00000000e+00,
			-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -3.47589890e-03, -1.79334304e-02, -6.26310905e-03, -0.00000000e+00

		};

		double expected_flow[70] =
		{
			-0.0254236,-0.0254236,-0.0198916,-0.0166246,-0.00827281,-0.00387269,-0.00387269,
			-0.0254236,-0.0254236,-0.0215234,-0.0173481,-0.0137003,-0.00827281,-0.00827281,
			-0.0262542,-0.0254236,-0.0215234,-0.0173481,-0.0121782,-0.00827281,-0.0073416,
			-0.0254236,-0.0239406,-0.0198916,-0.0173481,-0.0121782,-0.0073416,-0.0073416,
			-0.0254236,-0.0254236,-0.0198916,-0.0173481,-0.0137003,-0.0073416,-0.0073416,

			0.0098843,0.00982592,0.00746745,0.00258216,-0.00894369,-0.0171297,-0.0171297,
			0.0098843,0.00982592,0.00746745,0.00258216,-0.00894369,-0.0171297,-0.0171297,
			0.0098843,0.00982592,0.00452616,0.00258216,-0.00894369,-0.0171297,-0.0171297,
			0.0127247,0.00982592,0.00746745,0.00427295,-0.00894369,-0.0171297,-0.0171297,
			0.0127247,0.0098843,0.0078971,0.00452616,-0.00252711,-0.0171297,-0.0171297,
		};

		std::shared_ptr<DerivativeCalculator<double>> flow_deriv_calc
			= std::make_shared<DerivativeCalculator<double>>();
		double sigma_div = 0.3;
		double sigma_error = 20;
		double filter_influence = 5;
		double auxilary_influence = 3.921568627450981e-07;
		double sigma_distance = 7;
		double sigma_color = 0.027450980392156862;
		int filter_length = 15;

		std::array<const size_t, 3> img_shape = { 3,5,7 };
		auto ptr_image = std::make_shared<Array<float, 3>>(img_shape, image);

		std::array<const size_t, 3> flow_shape = { 2,5,7 };
		auto ptr_flow = std::make_shared<Array<double, 3>>(flow_shape, flow);
		auto ptr_flow_result = std::make_shared<Array<double, 3>>(flow_shape, flow);

		std::array<const size_t, 2> occ_shape = { 5,7 };
		auto ptr_log_occlusion = std::make_shared<Array<double, 2>>(occ_shape, log_occlusion);


		auto settings = std::make_shared<CrossMedianFilterSettings>();
		settings->SigmaDiv = sigma_div;
		settings->SigmaError = sigma_error;
		settings->FilterInfluence = filter_influence;
		settings->SigmaDistance = sigma_distance;
		settings->SigmaColor = sigma_color;
		settings->FilterLength = filter_length;
		settings->MedianFilterLength = 9;
		settings->Speedup = true;

		CrossBilateralMedianFilter filter(flow_deriv_calc, settings);
		filter.SetAuxiliaryInfluence(auxilary_influence);
		filter.SetCrossFilterImage(ptr_image);
		filter.ApplyTo(ptr_flow_result, ptr_flow);

		for (int i = 0; i < 70; i++)
		{
			EXPECT_NEAR(expected_flow[i], ptr_flow_result->Data()[i], 0.0001);
			if (abs(expected_flow[i] - ptr_flow_result->Data()[i]) > 0.0001)
			{
				printf("Test\n");
			}
		}
		
	}
}