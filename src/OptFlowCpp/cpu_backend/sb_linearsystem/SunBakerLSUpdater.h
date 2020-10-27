#pragma once
#include"optflow_solvers/linearsystems/ISunBakerLSUpdater.h"
#include"../Array.h"
#include"../image/inner/DerivativeCalculator.h"
#include"../Reshaper.h"
#include"core/penalty/IPenalty.h"
namespace cpu_backend
{
	class SunBakerLSUpdater : optflow_solvers::ISunBakerLSUpdater
	{
	public:
		using PtrGrayImg = std::shared_ptr<core::IArray<float, 2>>;
		using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
		// Inherited via ISunBakerLSBuilder
		virtual void SetFramePair(PtrGrayImg first_image, PtrGrayImg second_image) override;
		virtual void UpdateParameter(PtrFlowField linearization_points, double relaxation) override;
		virtual std::shared_ptr<core::ILinearProblem<double>> Update() override;
		virtual void SetPenalty(std::shared_ptr<core::IPenalty<double>> penalty) override;

		SunBakerLSUpdater(std::shared_ptr<DerivativeCalculator> deriv_calculator,

			double lambda_kernel);

		//D_y
		std::shared_ptr<Array<double, 1>> _data_y;

		//D_x
		std::shared_ptr<Array<double, 1>> _data_x;

		//R
		std::shared_ptr<Array<double, 1>> _rest;

		//b
		std::shared_ptr<Array<double, 1>> _desired_result;

	private:

		void initialize_deriv(size_t size);
		void initialize_data(size_t data_size);

		std::shared_ptr<DerivativeCalculator> _deriv_calculator;

		//I_x
		std::shared_ptr<Array<float, 1>> _image_deriv_x;
		//I_y
		std::shared_ptr<Array<float, 1>> _image_deriv_y;
		//I_t
		std::shared_ptr<Array<float, 1>> _image_deriv_time;

		

		std::shared_ptr<core::IPenalty<double>> _penalty;

		size_t _width = 0;
		size_t _height = 0;

		double _lambda_kernel;


		
	};
}