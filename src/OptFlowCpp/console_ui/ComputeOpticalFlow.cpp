#pragma once
#include"ComputeOpticalFlow.h"
#include<iostream>
#include"utilities/image_helper/ImageHelper.h"
#include"utilities/flow_helper/FlowHelper.h"
#include"Hypodermic/ContainerBuilder.h"
#include"setup/RegisterCGSolver.h"
#include"setup/RegisterCPUBackend.h"
#include"setup/RegisterSunBakerSolver.h"
#include"core/IArrayFactory.h"
#include"core/solver/problem/IProblemFactory.h"
#include"core/solver/IFlowFieldSolver.h"
#include"optflow_solvers/solvers/GNCPenaltySolver.h"
#include"optflow_solvers/solvers/LinearizationSolver.h"

namespace console_ui
{
    std::shared_ptr<Hypodermic::Container> SetupSolvers()
    {
        
        Hypodermic::ContainerBuilder builder;
        SetDefaultCGSettings(builder);
        RegisterCGSolver(builder);
        SetCPUBackendDefaultSettings(builder);
        RegisterCPUBackend(builder);
        SetDefaultSunBakerSettings(builder);
        RegisterSunBakerSolver(builder);

        return builder.build();
    }

    std::shared_ptr<core::IArray<float, 3>> OpenImage(std::string filepath, std::shared_ptr<Hypodermic::Container> container)
    {
        auto array_factory = container->resolve<core::IArrayFactory<float, 3>>();
        imagehelper::Image image = imagehelper::OpenImage(filepath);
        
        if (image.color_count < 3)
            return nullptr;
        image.color_count = 3;

        std::array<const size_t, 3> shape = {image.color_count, image.height,image.width };


        return array_factory->CreateFromSource(image.data->data(), shape);
    }

    void ComputeOpticalFlow(std::string first_image_path, std::string second_image_path, std::string flow_output_path, std::string flow_img_output_path)
    {
        auto di_container = SetupSolvers();
        auto first_img = OpenImage(first_image_path, di_container);
        auto second_img = OpenImage(second_image_path, di_container);

        if (first_img == nullptr || second_img == nullptr)
        {
            std::cout << "Error opening images" << std::endl;
            return;
        }
        

        auto problem_factory = di_container->resolve<core::IProblemFactory>();
        auto problem = problem_factory->CreateGrayCrossFilterProblem(first_img, second_img);

        //auto flow_solver = di_container->resolve<core::IFlowFieldSolver<std::shared_ptr<core::IGrayCrossFilterProblem>>>();
        
        auto temp = di_container->resolve< optflow_solvers::LinearizationSolver>();
        auto flow_solver = di_container->resolve < optflow_solvers::GNCPenaltySolver>();
        auto flow_field = flow_solver->Solve(problem);
        flowhelper::SaveFlow(flow_output_path, flow_field);
        flowhelper::SaveFlow2Color(flow_img_output_path, flow_field);
        std::cout << "Solving finished" << std::endl;
    }
}