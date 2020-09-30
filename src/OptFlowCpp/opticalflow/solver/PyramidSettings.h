#pragma once

namespace optflow
{
	namespace solver
	{
		template<class LayerSettingsTyp>
		class PyramidSettings
		{
		public:
			std::vector<double> ScaleFactors = { 0.5,0.5 };
			LayerSettingsTyp LayerSettings;
		};
	}
}