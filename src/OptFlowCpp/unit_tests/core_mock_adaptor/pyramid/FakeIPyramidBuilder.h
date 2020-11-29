#pragma once
#include"core/pyramid/IPyramidBuilder.h"
#include"core/solver/problem/IGrayPenaltyCrossProblem.h"
#include"gmock/gmock.h"

namespace core
{
	namespace testing
	{

		class FakeIPyramidBuilder : public IPyramidBuilder<IGrayPenaltyCrossProblem>
		{
		public:
			FakeIPyramidBuilder(std::shared_ptr<IPyramid<IGrayPenaltyCrossProblem>> pyramid);

			// Inherited via IPyramidBuilder
			MOCK_METHOD( void, SetScaleFactors ,(std::vector<double> factors),( override));

			MOCK_METHOD2(SetScaleFactor,
				void(double factor, std::array<size_t, 2> min_resolution));

			MOCK_METHOD1( SetResolutions ,
				void(std::vector<std::array<size_t, 2>> resolutions));

			


			// Inherited via IPyramidBuilder
			virtual std::shared_ptr<IPyramid<IGrayPenaltyCrossProblem>> Create(std::shared_ptr<IGrayPenaltyCrossProblem> last_level) override;
		private:
			
			std::shared_ptr<IPyramid<IGrayPenaltyCrossProblem>> _pyramid;
		};
	}
}