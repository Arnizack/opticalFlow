#pragma once
#include<string>
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

	

	void ComputeOpticalFlow(std::string first_image_path, std::string second_image_path, std::string flow_output_path, std::string flow_img_output_path);
}