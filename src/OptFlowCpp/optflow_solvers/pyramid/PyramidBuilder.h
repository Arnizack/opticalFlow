#pragma once
#include"core/pyramid/IPyramidBuilder.h"
#include"core/IScaler.h"

namespace optflow_solvers
{

	namespace _inner
	{
		enum class ResolutionDefinition
		{
			FACTORS, FACTORS_MINRES,RESOLUTIONS
		};
	}

	template<class InnerTyp>
	class PyramidBuilder : public core::IPyramidBuilder<InnerTyp>
	{
	public:
		PyramidBuilder(std::shared_ptr<core::IScaler<InnerTyp>> scaler);
		virtual void SetScaleFactors(std::vector<double> factors) override;
		virtual void SetScaleFactor(double factor, std::array<size_t, 2> min_resolution) override;
		//The original resolution should not be in resolutions
		virtual void SetResolutions(std::vector < std::array<size_t, 2>> resolutions) override;
		virtual std::shared_ptr<core::IPyramid<InnerTyp>> Create(std::shared_ptr<InnerTyp> last_level) = 0;

	protected:
		void ComputeResolutionFromFactors(size_t width, size_t height);
		void ComputeResolutionFromFactorMinRes(size_t width, size_t height);
		std::vector<std::shared_ptr<InnerTyp>> CreateLayers(std::shared_ptr<InnerTyp> last_layer);
		_inner::ResolutionDefinition _resolution_definition;

	private:

		
		std::shared_ptr<core::IScaler<InnerTyp>> _scaler;

		std::vector < std::array<size_t, 2>> _resolutions;
		std::vector<double> _factors;
		double _factor;
		std::array<size_t, 2> _min_resolution;

		


	};


	template<class InnerTyp>
	inline PyramidBuilder<InnerTyp>::PyramidBuilder(std::shared_ptr<core::IScaler<InnerTyp>> scaler)
		: _scaler(scaler)
	{
	}

	template<class InnerTyp>
	inline void PyramidBuilder<InnerTyp>::SetScaleFactors(std::vector<double> factors)
	{
		_resolution_definition = _inner::ResolutionDefinition::FACTORS;
		_factors = factors;
		
	}
	template<class InnerTyp>
	inline void PyramidBuilder<InnerTyp>::SetScaleFactor(double factor, std::array<size_t, 2> min_resolution)
	{
		_resolution_definition = _inner::ResolutionDefinition::FACTORS_MINRES;
		_factor = factor;
		_min_resolution = min_resolution;
	}
	template<class InnerTyp>
	inline void PyramidBuilder<InnerTyp>::SetResolutions(std::vector<std::array<size_t, 2>> resolutions)
	{
		_resolution_definition = _inner::ResolutionDefinition::RESOLUTIONS;
		_resolutions = resolutions;
	}

	template<class InnerTyp>
	inline void PyramidBuilder<InnerTyp>::ComputeResolutionFromFactors(size_t width, size_t height)
	{
		double width_d = width;
		double height_d = height;

		_resolutions.clear();
		

		for (double& factor : _factors)
		{
			width_d *= factor;
			height_d *= factor;
			_resolutions.push_back({ (size_t) width_d, (size_t) height_d });
		}
	}
	template<class InnerTyp>
	inline void PyramidBuilder<InnerTyp>::ComputeResolutionFromFactorMinRes(size_t width, size_t height)
	{
		
		double width_d = width;
		double height_d = height;

		_resolutions.clear();
		

		double min_width = _min_resolution[0];
		double min_height = _min_resolution[1];

		while (width_d > min_width && height_d > min_height)
		{
			width_d *= _factor;
			height_d *= _factor;
			_resolutions.push_back({ (size_t)width_d, (size_t)height_d });
		}
	}
	template<class InnerTyp>
	inline std::vector<std::shared_ptr<InnerTyp>> PyramidBuilder<InnerTyp>::CreateLayers(std::shared_ptr<InnerTyp> last_layer)
	{
		std::vector<std::shared_ptr<InnerTyp>> layers;
		layers.push_back(last_layer);
		for (auto resolution : _resolutions)
		{
			auto layer = _scaler->Scale(last_layer, resolution[0], resolution[1]);
			layers.push_back(layer);
		}
		return layers;
	}
}