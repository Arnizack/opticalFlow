#pragma once
#include "core/pyramid/IPyramidBuilder.h"
#include "core/solver/problem/IProblemFactory.h"
#include "core/IScaler.h"

#include "Pyramid.h"

namespace cpu_backend
{
	/*
	* ALL
	*/
	template<class T>
	class PyramidBuilder : public core::IPyramidBuilder<T>
	{
	public:
		PyramidBuilder(std::shared_ptr<core::IProblemFactory> problem_factory, std::shared_ptr<core::IScaler<T>> problem_scaler)
			: _resolutions({ {0,0} }), _factors({ 0 }), _problem_factory(problem_factory), _problem_scaler(problem_scaler)
		{}

		virtual void SetScaleFactors(std::vector<double> factors) override
		{
			/*
			*	1 > factor > 0
			*
			*	res calculated as next_res = current_res * current_factor
			*	not next_res = max_res * current_factor
			*/

			_factors = factors;

			_resolutions = { {0,0} };
			_min_resolution = { 0,0 };
		}

		virtual void SetScaleFactor(double factor, std::array<size_t, 2> min_resolution) override
		{
			/*
			* 1 > factor > 0
			*
			* calculated from max_res to min_res
			* by next_res = current_res * factor
			*/
			_resolutions = { {0,0} };

			_min_resolution = min_resolution;

			_factors = { factor };
		}

		virtual void SetResolutions(std::vector< std::array<size_t, 2>> resolutions) override
		{
			/*
			* resolutions sorted by biggest to smallest
			* a_1 > a_2 > ... > a_n
			*/
			_resolutions = resolutions;

			_factors = { 0 };
			_min_resolution = { 0,0 };
		}

		virtual std::shared_ptr<core::IPyramid<T>> Create(T last_level) override
		{
			//last_level has the biggest resolution
			//returns smallest resolution to biggest

			setup_resolutions(last_level);

			const size_t length = _resolutions.size();

			std::vector<T> levels(length); //max to min

			levels[0] = last_level;

			for (size_t i = 1; i < length; i++)
			{
				levels[i] = _problem_scaler->Scale(levels[i - 1], _resolutions[i][0], _resolutions[i][0]);
			}

			return nullptr;
		}

	private:

		void setup_resolutions(T last_level)
		{
			if (_resolutions.size() == 1 && _resolutions[0][0] == 0 && _resolutions[0][1] == 0 && _min_resolution[0] == 0 && _min_resolution[0] == 0)
			{
				//SetScaleFactors

				const size_t size = _factors.size();
				const size_t width = last_level->FirstFrame->Shape[0]; //max
				const size_t height = last_level->FirstFrame->Shape[1];

				_resolutions = std::vector<std::array<size_t, 2>>(size + 1);

				_resolutions[0] = { width, height }; //max

				for (size_t i = 0; i < size; i++)
				{
					_resolutions[i + 1] = { (size_t)(_resolutions[i][0] * _factors[i]), (size_t)(_resolutions[i][1] * _factors[i]) };
				}
				return;
			}
			else if (_factors.size() == 1 && _factors[0] == 0)
			{
				//SetResolutions
				return;
			}
			else
			{
				//SetScaleFactor

				size_t in_width = last_level->FirstFrame->Shape[0]; //max
				size_t in_height = last_level->FirstFrame->Shape[1];

				double factor = _factors[0]; // 1>factor>0

				size_t min_width = _min_resolution[0]; // min
				size_t min_height = _min_resolution[1];

				_resolutions[0] = { in_width, in_height };

				size_t iter_width = _resolutions[0][0] * factor;
				size_t iter_height = _resolutions[0][1] * factor;

				while (iter_width > min_width && iter_height > min_height)
				{
					_resolutions.push_back({ iter_width, iter_height });

					iter_width *= factor;
					iter_height *= factor;
				}
				_resolutions.push_back({ min_width, min_height });

				return;
			}
		}

		//Member
		std::vector< std::array<size_t, 2>> _resolutions; //max to min, first_res = max_res
		std::vector<double> _factors;
		std::array<size_t, 2> _min_resolution;

		//factories and operations
		std::shared_ptr<core::IProblemFactory> _problem_factory;
		std::shared_ptr<core::IScaler<T>> _problem_scaler;
	};


	/*
	* I GRAY PENALTY CROSS PROBLEM
	*/
	template<>
	class PyramidBuilder<std::shared_ptr<core::IGrayPenaltyCrossProblem>> : public core::IPyramidBuilder< std::shared_ptr<core::IGrayPenaltyCrossProblem>>
	{
		using PyramidIGrayPenaltyCrossProblem = Pyramid<std::shared_ptr<core::IGrayPenaltyCrossProblem>>;
		using PtrIGrayPenaltyCrossProblem = std::shared_ptr<core::IGrayPenaltyCrossProblem>;


	public:
		PyramidBuilder(std::shared_ptr<core::IProblemFactory> problem_factory, std::shared_ptr<core::IScaler<core::IGrayPenaltyCrossProblem>> problem_scaler)
			: _resolutions({ {0,0} }), _factors({ 0 }), _min_resolution({0,0}), _problem_factory(problem_factory), _problem_scaler(problem_scaler)
		{}

		virtual void SetScaleFactors(std::vector<double> factors) override
		{
			/* 
			*	1 > factor > 0
			* 
			*	res calculated as next_res = current_res * current_factor
			*	not next_res = max_res * current_factor 
			*/

			_factors = factors;

			_resolutions = { {0,0} };
			_min_resolution = { 0,0 };
		}

		virtual void SetScaleFactor(double factor, std::array<size_t, 2> min_resolution) override
		{
			/*
			* 1 > factor > 0
			* 
			* calculated from max_res to min_res
			* by next_res = current_res * factor
			*/
			_resolutions = { {0,0} };

			_min_resolution = min_resolution;

			_factors = { factor };
		}

		virtual void SetResolutions(std::vector< std::array<size_t, 2>> resolutions) override
		{
			/*
			* resolutions sorted by biggest to smallest
			* a_1 > a_2 > ... > a_n
			*/
			_resolutions = resolutions;

			_factors = { 0 };
			_min_resolution = { 0,0 };
		}

		virtual std::shared_ptr<core::IPyramid<PtrIGrayPenaltyCrossProblem>> Create(PtrIGrayPenaltyCrossProblem last_level) override
		{
			//last_level has the biggest resolution
			//returns smallest resolution to biggest
			
			setup_resolutions(last_level);

			const size_t length = _resolutions.size();

			std::vector<PtrIGrayPenaltyCrossProblem> levels(length); //max to min

			levels[0] = last_level;

			for (size_t i = 1; i < length; i++)
			{
				levels[i] = _problem_scaler->Scale(levels[i - 1], _resolutions[i][0], _resolutions[i][0]);
			}

			return std::make_shared<PyramidIGrayPenaltyCrossProblem>(PyramidIGrayPenaltyCrossProblem(levels));
		}

	private:

		void setup_resolutions(PtrIGrayPenaltyCrossProblem last_level)
		{
			if (_resolutions.size() == 1 && _resolutions[0][0] == 0 && _resolutions[0][1] == 0 && _min_resolution[0] == 0 && _min_resolution[0] == 0 && _factors.size() > 1)
			{
				//SetScaleFactors

				const size_t size = _factors.size();
				const size_t width = last_level->FirstFrame->Shape[0]; //max
				const size_t height = last_level->FirstFrame->Shape[1];

				_resolutions = std::vector<std::array<size_t, 2>>(size + 1);

				_resolutions[0] = { width, height }; //max

				for (size_t i = 0; i < size; i++)
				{
					_resolutions[i + 1] = { (size_t)(_resolutions[i][0] * _factors[i]), (size_t)(_resolutions[i][1] * _factors[i]) };
				}
				return;
			}
			else if (_factors.size() == 1 && _factors[0] == 0)
			{
				//SetResolutions
				return;
			}
			else
			{
				//SetScaleFactor

				size_t in_width = last_level->FirstFrame->Shape[0]; //max
				size_t in_height = last_level->FirstFrame->Shape[1];

				double factor = _factors[0]; // 1>factor>0

				size_t min_width = _min_resolution[0]; // min
				size_t min_height = _min_resolution[1];

				_resolutions[0] = { in_width, in_height };

				size_t iter_width = _resolutions[0][0] * factor;
				size_t iter_height = _resolutions[0][1] * factor;
				
				while (iter_width > min_width && iter_height > min_height)
				{
					_resolutions.push_back( { iter_width, iter_height } );

					iter_width *= factor;
					iter_height *= factor;
				}
				_resolutions.push_back({ min_width, min_height });

				return;
			}
		}

		//Member
		std::vector< std::array<size_t, 2>> _resolutions; //max to min, first_res = max_res
		std::vector<double> _factors;
		std::array<size_t, 2> _min_resolution;

		//factories and operations
		std::shared_ptr<core::IProblemFactory> _problem_factory;
		std::shared_ptr<core::IScaler<core::IGrayPenaltyCrossProblem>> _problem_scaler;
	};
}