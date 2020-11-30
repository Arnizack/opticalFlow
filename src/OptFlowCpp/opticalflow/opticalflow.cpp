#include"opticalflow.h"
#include"image_helper/ImageHelper.h"
#include"flow_helper/FlowHelper.h"
#include"json_settings/JSONHandler.h"
#include"core/solver/problem/IProblemFactory.h"
#include"core/solver/IFlowFieldSolver.h"
#include"core/solver/problem/IGrayCrossFilterProblem.h"
#include"core/IArrayFactory.h"
#include"optflow_composition/ContainerInstaller.h"
#include"Base.h"
#include"Hypodermic/ContainerBuilder.h"
#include"optflow_solvers/solvers/GNCPenaltySolver.h"
namespace opticalflow
{

    flowhelper::FlowField FlowFieldToHelperFlowField(FlowField flow)
    {
        flowhelper::FlowField helper_flow;
        helper_flow.height = flow.height;
        helper_flow.width = flow.width;
        helper_flow.data = flow.data;
        return helper_flow;
    }

    void SaveFlowField(std::string filepath, FlowField flow) 
    {
        flowhelper::SaveFlow(filepath,FlowFieldToHelperFlowField(flow));
    }
    
    void SaveFlowFieldToColor(std::string filepath, FlowField flow) 
    {
        flowhelper::SaveFlow2Color(filepath,FlowFieldToHelperFlowField(flow));
    }

    Image OpenImage(std::string filepath) 
    {
        auto helper_img = imagehelper::OpenImage(filepath);
        Image img;
        img.color_bands = helper_img.color_count;
        img.width = helper_img.width;
        img.height = helper_img.height;
        img.data = helper_img.data;
        return img;
    }
    
    std::shared_ptr<SolverOptions> ReadOptions(std::string filepath) 
    {
        auto options = std::make_shared<optflow_composition::ContainerOptions>();

        //Parse JSON
        json_settings::JsonSetupSettings(filepath, options);
        
        return options;
    }

    class SunBakerOpticalFlowSolver : public OpticalFlowSolver
    {
    private:
        std::shared_ptr< core::IArrayFactory<float, 3>> _image_factory;
		std::shared_ptr< core::IProblemFactory> _problem_factory;
		std::shared_ptr< core::IFlowFieldSolver < std::shared_ptr<core::IGrayCrossFilterProblem>>> _flow_solver;
    public:
        SunBakerOpticalFlowSolver(std::shared_ptr< core::IArrayFactory<float, 3>> image_factory,
		std::shared_ptr< core::IProblemFactory> problem_factory,
            std::shared_ptr< optflow_solvers::GNCPenaltySolver> flow_solver)
            : _image_factory(image_factory), _problem_factory(problem_factory),_flow_solver(flow_solver)
        {
            //OPF_LOG_IMAGE_FLOW_BEGIN();
            
        };
        virtual FlowField Solve(Image first_frame, Image second_frame)
        {
            OPF_PROFILE_BEGIN_SESSION("solvers", "solvers_profile.json");
            OPF_PROFILE_FUNCTION();
            size_t width = first_frame.width;
            size_t height = first_frame.height;
            std::array<const size_t,3> img_shape = {first_frame.color_bands,height,width};
            auto array_first_img = _image_factory->CreateFromSource(first_frame.data->data(),img_shape);
            auto array_second_img = _image_factory->CreateFromSource(second_frame.data->data(),img_shape);
            auto problem = _problem_factory->CreateGrayCrossFilterProblem(array_first_img,array_second_img);
            auto array_flow = _flow_solver->Solve(problem);
            
            FlowField result_flow;
            result_flow.width = width;
            result_flow.height = height;
            result_flow.data = std::make_shared<std::vector<double>>(width*height*2);
            array_flow->CopyDataTo(result_flow.data->data());
            OPF_PROFILE_END_SESSION();
            return result_flow;
        }

        ~SunBakerOpticalFlowSolver()
        {
            OPF_LOG_IMAGE_FLOW_END();
            
        }        

    };

    std::shared_ptr<Hypodermic::Container> SetupDIContainer(std::shared_ptr<SolverOptions> options)
    {
        optflow_composition::ContainerInstaller di_installer;
        di_installer.SetOptions(options);
        auto solver_container = di_installer.Install();
        Hypodermic::ContainerBuilder builder;
        builder.registerType<SunBakerOpticalFlowSolver>();
        return builder.buildNestedContainerFrom(*solver_container);
    }

    std::shared_ptr<OpticalFlowSolver> CreateSolver(std::shared_ptr<SolverOptions> options)
    {
        core::Logger::Init();
        debug_helper::ImageLogger::Init(
            //"E:\\dev\\opticalFlow\\optFlowCpp\\opticalFlow\\src\\OptFlowCpp\\bin\\debug_images"
            "debug_images",
            //"E:\\dev\\opticalFlow\\optFlowCpp\\opticalFlow\\src\\OptFlowCpp\\bin\\debug_flow"
            "debug_flow");

	    OPF_LOG_INFO("Start Logger");
        OPF_PROFILE_BEGIN_SESSION("dependency injection","dependency_injection_profile.json");
        std::shared_ptr<SunBakerOpticalFlowSolver> solver;
        {
            OPF_PROFILE_SCOPE("dependency injection setup");
            auto container = SetupDIContainer(options);
            solver = container->resolve<SunBakerOpticalFlowSolver>();
        }
        OPF_PROFILE_END_SESSION();
        return solver;

    }
}