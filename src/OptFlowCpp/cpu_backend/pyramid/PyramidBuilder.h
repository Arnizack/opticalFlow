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
		PyramidBuilder(std::shared_ptr<core::IProblemFactory> problem_factory, std::shared_ptr<core::IScaler<float, 2>> scaler_2D,
			std::shared_ptr<core::IScaler<float, 3>> scaler_3D)
			: _resolutions({ {0,0} }), _factors({ 0 }), _problem_factory(problem_factory), _scaler_2D(scaler_2D), _scaler_3D(scaler_3D)
		{}

		virtual void SetScaleFactors(std::vector<double> factors) override
		{
			//factors sorted from biggest to smallest
			_factors = factors;

			_resolutions = { {0,0} };
		}

		virtual void SetScaleFactor(double factor, std::array<size_t, 2> min_resolution) override
		{
			//factor scales min_resolution
			_resolutions = { min_resolution };

			_factors = { factor };
		}

		virtual void SetResolutions(std::vector< std::array<size_t, 2>> resolutions) override
		{
			//resolutions sorted from biggest to smallest
			_resolutions = resolutions;

			_factors = { 0 };
		}

		virtual std::shared_ptr<core::IPyramid<T>> Create(T last_level) override
		{
			return nullptr;
		}

	private:

		void setup_resolutions(T last_level)
		{
			if (_resolutions.size() == 1 && _resolutions[0][0] == 0 && _resolutions[0][1] == 0)
			{
				const size_t size = _factors.size();

				_resolutions = std::vector<std::array<size_t, 2>>(size);

				for (size_t i = 0; i < size; i++)
				{
					_resolutions[i] = { last_level->FirstFrame->Shape[0] * _factors[i], last_level->FirstFrame->Shape[1] * _factors[i] };
				}
				return;
			}
			else if (_factors.size() == 1 && _factors[0] == 0)
			{
				return;
			}
			else
			{
				size_t* back_width = &_resolutions.back()[0];
				size_t* back_height = &_resolutions.back()[1];

				*back_width *= _factors[0];
				*back_height *= _factors[0];

				while (*back_width <= last_level->FirstFrame->Shape[0] && *back_height <= last_level->FirstFrame->Shape[1])
				{
					_resolutions.push_back({ *back_width, *back_height });

					*back_width *= _factors[0];
					*back_height *= _factors[0];
				}
				return;
			}
		}

		//Member
		std::vector< std::array<size_t, 2>> _resolutions;
		std::vector<double> _factors;

		//factories and operations
		std::shared_ptr<core::IProblemFactory> _problem_factory;
		std::shared_ptr<core::IScaler<float, 2>> _scaler_2D;
		std::shared_ptr<core::IScaler<float, 3>> _scaler_3D;
	};


	/*
	* I GRAY PENALTY CROSS PROBLEM
	*/
	template<>
	class PyramidBuilder<std::shared_ptr<core::IGrayPenaltyCrossProblem>> : public core::IPyramidBuilder< std::shared_ptr<core::IGrayPenaltyCrossProblem>>
	{
		using PtrArray = std::shared_ptr<core::IArray<double, 2>>;
		using PtrPyramidIGrayPenaltyCrossProblem = std::shared_ptr<Pyramid<std::shared_ptr<core::IGrayPenaltyCrossProblem>>>;
		using PyramidIGrayPenaltyCrossProblem = Pyramid<std::shared_ptr<core::IGrayPenaltyCrossProblem>>;
		using PtrIGrayPenaltyCrossProblem = std::shared_ptr<core::IGrayPenaltyCrossProblem>;


	public:
		PyramidBuilder(std::shared_ptr<core::IProblemFactory> problem_factory, std::shared_ptr<core::IScaler<float, 2>> scaler_2D,
			std::shared_ptr<core::IScaler<float, 3>> scaler_3D)
			: _resolutions( {{0,0}}), _factors( {0} ), _problem_factory(problem_factory), _scaler_2D(scaler_2D), _scaler_3D(scaler_3D)
		{}

		virtual void SetScaleFactors(std::vector<double> factors) override
		{
			//factors sorted from biggest to smallest
			_factors = factors;

			_resolutions = { {0,0} };
		}

		virtual void SetScaleFactor(double factor, std::array<size_t, 2> min_resolution) override
		{
			//factor scales min_resolution
			_resolutions = { min_resolution };

			//_min_resolution = min_resolution;

			_factors = { factor };
		}

		virtual void SetResolutions(std::vector< std::array<size_t, 2>> resolutions) override
		{
			//resolutions sorted from biggest to smallest
			_resolutions = resolutions;

			_factors = { 0 };
		}

		virtual std::shared_ptr<core::IPyramid<PtrIGrayPenaltyCrossProblem>> Create(PtrIGrayPenaltyCrossProblem last_level) override
		{
			//last_level has the biggest resolution
			//returns smallest resolution to biggest
			
			setup_resolutions(last_level);

			auto lowest_res = _problem_factory->CreateGrayPenaltyCrossProblem();
			lowest_res->FirstFrame = _scaler_2D->Scale(last_level->FirstFrame, _resolutions[0][0], _resolutions[0][1]);
			lowest_res->SecondFrame = _scaler_2D->Scale(last_level->SecondFrame, _resolutions[0][0], _resolutions[0][1]);
			lowest_res->CrossFilterImage = _scaler_3D->Scale(last_level->CrossFilterImage, _resolutions[0][0], _resolutions[0][1]);
			lowest_res->PenaltyFunc = last_level->PenaltyFunc;

			PtrPyramidIGrayPenaltyCrossProblem pyramid = std::make_shared<PyramidIGrayPenaltyCrossProblem>(PyramidIGrayPenaltyCrossProblem(_resolutions, lowest_res, _problem_factory, _scaler_2D, _scaler_3D));

			return pyramid;
		}

	private:

		void setup_resolutions(PtrIGrayPenaltyCrossProblem last_level)
		{
			if (_resolutions.size() == 1 && _resolutions[0][0] == 0 && _resolutions[0][1] == 0)
			{
				//SetScaleFactors

				const size_t size = _factors.size();
				const size_t width = last_level->FirstFrame->Shape[0];
				const size_t height = last_level->FirstFrame->Shape[1];

				_resolutions = std::vector<std::array<size_t, 2>>(size);

				for (size_t i = 0; i < size; i++)
				{
					_resolutions[i] = { (size_t)(width * _factors[i]), (size_t)(height * _factors[i]) };
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

				size_t in_width = last_level->FirstFrame->Shape[0];
				size_t in_height = last_level->FirstFrame->Shape[1];

				double factor = _factors[0];

				size_t iter_width = _resolutions[0][0] * factor;
				size_t iter_height = _resolutions[0][1] * factor;
				
				while (iter_width < in_width && iter_height < in_height)
				{
					_resolutions.push_back( { iter_width, iter_height } );

					iter_width *= factor;
					iter_height *= factor;
				}
				_resolutions.push_back({ in_width, in_height });

				return;
			}
		}

		//Member
		std::vector< std::array<size_t, 2>> _resolutions;
		std::vector<double> _factors;

		//factories and operations
		std::shared_ptr<core::IProblemFactory> _problem_factory;
		std::shared_ptr<core::IScaler<float, 2>> _scaler_2D;
		std::shared_ptr<core::IScaler<float, 3>> _scaler_3D;
		//std::array<size_t, 2> _min_resolution;
	};
}