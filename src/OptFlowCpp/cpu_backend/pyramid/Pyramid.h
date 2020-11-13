#pragma once
#include <memory>
#include <vector>
#include <array>
#include "core/pyramid/IPyramid.h"
#include "core/solver/problem/IProblemFactory.h"
#include "core/IScaler.h"

namespace cpu_backend
{
	template<class T>
	class Pyramid : public core::IPyramid<T>
	{
	public:
		Pyramid(std::vector< std::array<size_t, 2>> resolutions, const T lowest_res,
			std::shared_ptr<core::IProblemFactory> problem_factory,	std::shared_ptr<core::IScaler<float, 2>> scaler_2D, 
			std::shared_ptr<core::IScaler<float, 3>> scaler_3D)
			: _resolutions(resolutions), _lowest_res(lowest_res), _end_level(resolutions.size() - 1), iter(0),
			_problem_factory(problem_factory), _scaler_2D(scaler_2D), _scaler_3D(scaler_3D)
		{}

		virtual T NextLevel() override
		{
			const size_t target_width = _resolutions[++iter][0];
			const size_t target_height = _resolutions[iter][1];

			auto temp_problem = _problem_factory->CreateGrayPenaltyCrossProblem();
			temp_problem->FirstFrame = _scaler_2D->Scale(_lowest_res->FirstFrame, target_width, target_height);
			temp_problem->SecondFrame = _scaler_2D->Scale(_lowest_res->SecondFrame, target_width, target_height);
			temp_problem->CrossFilterImage = _scaler_3D->Scale(_lowest_res->CrossFilterImage, target_width, target_height);
			temp_problem->PenaltyFunc = _lowest_res->PenaltyFunc;

			return temp_problem;
		}

		virtual bool IsEndLevel() override
		{
			if (iter == _end_level)
				return true;
			return false;
		}

	private:
		size_t iter;
		size_t _end_level;
		std::vector< std::array<size_t, 2>> _resolutions;
		const T _lowest_res;

		std::shared_ptr<core::IProblemFactory> _problem_factory;
		std::shared_ptr<core::IScaler<float, 2>> _scaler_2D;
		std::shared_ptr<core::IScaler<float, 3>> _scaler_3D;
	};
}