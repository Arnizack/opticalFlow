#pragma once
#include"ProgramOptions.h"

namespace console_ui
{
	namespace bpo = boost::program_options;

	std::shared_ptr<Hypodermic::Container> SetupCommandlineSolvers(boost::program_options::variables_map vm)
	{

		Hypodermic::ContainerBuilder builder;
		//SetCommandlineCGSettings(builder, vm);
		//RegisterCGSolver(builder);
		//SetCPUBackendCommandlineSettings(builder, vm);
		//RegisterCPUBackend(builder);
		//SetCommandlineSunBakerSettings(builder, vm);
		//RegisterSunBakerSolver(builder);

		return builder.build();
	}

	bool CheckCommandLineInput(int argc, char* argv[], std::string& first_image_path, std::string& second_image_path, std::string& flow_output_path, std::string& flow_img_output_path, std::string& json_input_path)
	{
		bpo::options_description generic_opt = SetupGenericOptions();

		bpo::options_description io_opt = SetupIOOptions(first_image_path, second_image_path, flow_output_path, flow_img_output_path, json_input_path);

		/*bpo::options_description cpu_backend_opt = SetupCPUBackendOptions();

		bpo::options_description cg_opt = SetupCGSolverOptions();

		bpo::options_description cpu_linalg_opt = SetupCPULinalgOptions();

		bpo::options_description sun_baker_solver_opt = SetupSunBakerSolverOptions();*/

		bpo::options_description cmd_opt("All Options");
		cmd_opt.add(generic_opt).add(io_opt);//.add(cpu_backend_opt).add(cg_opt).add(cpu_linalg_opt).add(sun_baker_solver_opt);

		bpo::variables_map input = ParseCommandline(argc, argv, cmd_opt);

		if (input.count("help"))
		{
			std::cout << cmd_opt << '\n';
			return false;
		}

		return true;
	}

	bpo::variables_map ParseCommandline(int argc, char* argv[], bpo::options_description options)
	{
		bpo::variables_map vm;
		bpo::store(bpo::parse_command_line(argc, argv, options), vm);
		bpo::notify(vm);

		return vm;
	}
}