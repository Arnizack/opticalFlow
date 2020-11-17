#pragma once
#include"ComputeOpticalFlow.h"
#include"setup/solvers/GNCSolverContainer.h"

namespace console_ui
{
    std::shared_ptr<Hypodermic::Container> SetupDefaultSolvers()
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

    std::shared_ptr<Hypodermic::Container> SetupDefaultSolvers2()
    {
        Hypodermic::ContainerBuilder builder;
        SetCPUBackendDefaultSettings(builder);
        RegisterCPUBackend(builder);
        auto cpu_backend = builder.build();

        Backends backends(cpu_backend);
        GNCSolverSettings settings;
        return GNCSolverContainer(backends, settings);
   
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

    void ComputeOpticalFlow(std::string first_image_path, std::string second_image_path, std::string flow_output_path, std::string flow_img_output_path, 
        std::shared_ptr<Hypodermic::Container> di_container)
    {
        //auto di_container = SetupDefaultSolvers();
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
        
        //auto temp = di_container->resolve< optflow_solvers::LinearizationSolver>();
        auto flow_solver = di_container->resolve < optflow_solvers::GNCPenaltySolver>();
        auto flow_field = flow_solver->Solve(problem);
        flowhelper::SaveFlow(flow_output_path, flow_field);
        flowhelper::SaveFlow2Color(flow_img_output_path, flow_field);
        std::cout << "Solving finished" << std::endl;
    }
}