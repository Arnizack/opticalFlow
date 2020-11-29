#pragma once
#include"core/IArray.h"
#include"core/IArrayFactory.h"
#include<memory>
#include"image_helper/ImageHelper.h"
#include<iostream>
#include"core/solver/problem/IProblemFactory.h"
#include"core/solver/IFlowFieldSolver.h"
#include"core/solver/problem/IGrayCrossFilterProblem.h"


namespace optflow_solvers
{
	class OpticalFlowApplication
	{
    private:
        std::shared_ptr< core::IArrayFactory<float, 3>> _image_factory;
        std::shared_ptr< core::IProblemFactory> _problem_factory;
        std::shared_ptr< core::IFlowFieldSolver < std::shared_ptr<core::IGrayCrossFilterProblem>>> _flow_solver;
	public:

		OpticalFlowApplication(std::shared_ptr< core::IArrayFactory<float, 3>> image_factory,
		std::shared_ptr< core::IProblemFactory> problem_factory,
			std::shared_ptr< core::IFlowFieldSolver < std::shared_ptr<core::IGrayCrossFilterProblem>>> flow_solver);

		std::shared_ptr<core::IArray<float, 3>> OpenImage(std::string filepath);

		void ComputeOpticalFlow(std::string first_image_path, std::string second_image_path, std::string flow_output_path, std::string flow_img_output_path);
	};
}