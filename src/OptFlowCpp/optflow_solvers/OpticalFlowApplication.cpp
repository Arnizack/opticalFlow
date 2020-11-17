#pragma once
#include"pch.h"
#include"OpticalFlowApplication.h"
#include"utilities/flow_helper/FlowHelper.h"

namespace optflow_solvers
{
    OpticalFlowApplication::OpticalFlowApplication(std::shared_ptr<core::IArrayFactory<float, 3>> image_factory, std::shared_ptr<core::IProblemFactory> problem_factory, std::shared_ptr<core::IFlowFieldSolver<std::shared_ptr<core::IGrayCrossFilterProblem>>> flow_solver)
        : _image_factory(image_factory), _problem_factory(problem_factory), _flow_solver(flow_solver)
    {}
    std::shared_ptr<core::IArray<float, 3>> OpticalFlowApplication::OpenImage(std::string filepath)
    {
        imagehelper::Image image = imagehelper::OpenImage(filepath);

        if (image.color_count < 3)
            return nullptr;
        image.color_count = 3;

        std::array<const size_t, 3> shape = { image.color_count, image.height,image.width };


        return _image_factory->CreateFromSource(image.data->data(), shape);
    }
    void OpticalFlowApplication::ComputeOpticalFlow(std::string first_image_path, std::string second_image_path, std::string flow_output_path, std::string flow_img_output_path)
    {
        //auto di_container = SetupDefaultSolvers();
        auto first_img = OpenImage(first_image_path);
        auto second_img = OpenImage(second_image_path);

        if (first_img == nullptr || second_img == nullptr)
        {
            std::cout << "Error opening images" << std::endl;
            return;
        }


        auto problem = _problem_factory->CreateGrayCrossFilterProblem(first_img, second_img);

        auto flow_field = _flow_solver->Solve(problem);
        flowhelper::SaveFlow(flow_output_path, flow_field);
        flowhelper::SaveFlow2Color(flow_img_output_path, flow_field);
        std::cout << "Solving finished" << std::endl;
    }
}