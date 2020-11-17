#pragma once
#include"SetupOptions.h"

namespace console_ui
{
	namespace bpo = boost::program_options;

	boost::program_options::options_description SetupGenericOptions()
	{
		bpo::options_description generic("Generic options");
		generic.add_options()
			("help", "produce help message");

		return generic;
	}

	boost::program_options::options_description SetupIOOptions(std::string first_image_path, std::string second_image_path, std::string flow_output_path, std::string flow_img_output_path)
	{
		bpo::options_description io("Input-Output options");
		io.add_options()
			("input-img1,1", bpo::value< std::string>(&first_image_path), "path to first input image")
			("input-img2,2", bpo::value< std::string>(&second_image_path), "path to second input image")
			("output-path,O", bpo::value< std::string>(&flow_output_path), "Flow output path")
			("flow-img,F", bpo::value< std::string>(&flow_img_output_path), "path to flow image");

		return io;
	}

	boost::program_options::options_description SetupCGSolverOptions()
	{
		bpo::options_description cg("Conjugate Gradient Solver options");
		cg.add_options()
			("cg_tol", bpo::value< double>()->default_value(1e-3), "Sets the tolerance for the Conjugate Gradient Solver")
			("cg_iter", bpo::value < size_t>()->default_value(100), "Sets the number of iterations for the Conjugate Gradient Solver");

		return cg;
	}

	boost::program_options::options_description SetupCPUBackendOptions()
	{
		bpo::options_description cpu_back("CPU Backend options");

		bpo::options_description penalty("Charbonnier Penalty function options");
		penalty.add_options()
			("penalty_blend", bpo::value<double>()->default_value(0), "Sets the default Blend-Factor of the Charbonnier Penalty function")
			("penalty_eps", bpo::value<double>()->default_value(0.001), "Sets the default Epsilon value of the Charbonnier Penalty function")
			("penalty_exp", bpo::value<double>()->default_value(0.45), "Sets the default Exponent of the Charbonnier Penalty function");

		bpo::options_description cross_median("Cross Median Filter options");
		cross_median.add_options()
			("cross_sig_div", bpo::value<double>()->default_value(0.3), "Sets the default Sigma-Division value of the Cross Median Filter")
			("cross_sig_err", bpo::value<double>()->default_value(20), "Sets the default Sigma-Error value of the Cross Median Filter")
			("cross_sig_dist", bpo::value<double>()->default_value(7), "Sets the default Sigma-Distance value of the Cross Median Filter")
			("cross_sig_col", bpo::value<double>()->default_value(7.0 / 255), "Sets the default Sigma-Color value of the Cross Median Filter")
			("cross_filt", bpo::value<double>()->default_value(5), "Sets the default Filter-Influence value of the Cross Median Filter")
			("cross_filt_len", bpo::value<int>()->default_value(15), "Sets the default Filter-Length value of the Cross Median Filter");

		bpo::options_description linear("Linear System options");
		linear.add_options()
			("lin_sys_lamb", bpo::value<double>()->default_value(10.0 / 255.0), "Sets the default Lambda-Kernel value of the Linear System");

		cpu_back.add(penalty).add(cross_median).add(linear);

		return cpu_back;
	}

	boost::program_options::options_description SetupCPULinalgOptions()
	{
		bpo::options_description cpu_linalg("CPU Linalg options");
		cpu_linalg.add_options();

		return cpu_linalg;
	}

	boost::program_options::options_description SetupSunBakerSolverOptions()
	{
		bpo::options_description sun_baker("Sun and Baker Solver options");

		bpo::options_description linearization("Linearization Solver options");
		linearization.add_options()
			("lin_start", bpo::value<double>()->default_value(1e-04 / 255.0), "Sets the default Start-Relaxation value of the Linearization Solver")
			("lin_end", bpo::value<double>()->default_value(1e-01 / 255.0), "Sets the default End-Relaxation value of the Linearization Solver")
			("lin_steps", bpo::value<double>()->default_value(3), "Sets the default Relaxation-Steps value of the Linearization Solver");

		bpo::options_description incremental("Incremental Solver options");
		incremental.add_options()
			("inc_steps", bpo::value<int>()->default_value(3), "Sets the default number of Steps of the Incremental Solver");

		bpo::options_description gnc("GNC Penalty Solver options");
		gnc.add_options()
			("gnc_steps", bpo::value<int>()->default_value(3), "Sets the default number of Steps of the GNC Penalty Solver");

		sun_baker.add(linearization).add(incremental).add(gnc);

		return sun_baker;
	}
}