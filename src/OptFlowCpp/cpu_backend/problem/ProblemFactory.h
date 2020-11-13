#pragma once
#include "core/solver/problem/IProblemFactory.h"
#include"../ArrayFactory.h"
#include<memory>

namespace cpu_backend
{
	class ProblemFactory : public core::IProblemFactory
	{
	public:
		ProblemFactory(std::shared_ptr<ArrayFactory<float,2>> grayscale_factory);
		virtual std::shared_ptr<core::IGrayPenaltyCrossProblem> CreateGrayPenaltyCrossProblem() override
		{
			return std::make_shared<core::IGrayPenaltyCrossProblem>( core::IGrayPenaltyCrossProblem() );
		}

		virtual std::shared_ptr<core::IGrayCrossFilterProblem> CreateGrayCrossFilterProblem(std::shared_ptr<core::IArray<float, 3>> first_image, std::shared_ptr<core::IArray<float, 3>> seconde_image) override;
	
	private:
		std::shared_ptr<ArrayFactory<float, 2>> _grayscale_factory;

	};
}