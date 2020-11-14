#pragma once
#include"RegisterCPUBackend.h"
#include"cpu_backend/penalty/CharbonnierPenalty.h"
#include"cpu_backend/flow/CrossBilateralMedianFilter.h"
#include"cpu_backend/image/warper/GrayWarper.h"
#include"cpu_backend/image/inner/DerivativeCalculator.h"

//#include"cpu_backend/pyramid/Pyramid.h"
#include"cpu_backend/problem/ProblemFactory.h"
#include"cpu_backend/penalty/CharbonnierPenalty.h"
#include"cpu_backend/sb_linearsystem/SunBakerLSUpdater.h"
#include"cpu_backend/ArrayScaler.h"
#include"cpu_backend/flow/FlowFieldScaler.h"
#include"cpu_backend/problem/GrayPenaltyCrossProblemScaler.h"
#include"RegisterCPULinalg.h"

namespace console_ui
{
    void RegisterCPUBackend(Hypodermic::ContainerBuilder& builder)
    {
        RegisterCPULinalg(builder);

        //Penalty
        builder.registerType<cpu_backend::CharbonnierPenalty>()
            .as<core::IBlendablePenalty<double>>();

        //Cross Filter
        builder.registerType<cpu_backend::CrossBilateralMedianFilter>()
            .as<core::ICrossFlowFilter>()
            .singleInstance();

        //Derivative Calculator
        builder.registerType<cpu_backend::DerivativeCalculator<double>>()
            .singleInstance();

        

        //Scaler core::IScaler<float, 2>
        builder.registerType<cpu_backend::ArrayScaler<float, 2>>()
            .as<core::IScaler<core::IArray<float, 2>>>()
            .singleInstance();

        //Scaler core::IScaler<float, 3>
        builder.registerType<cpu_backend::ArrayScaler<float, 3>>()
            .as<core::IScaler<core::IArray<float, 3>>>()
            .singleInstance();
        //Scaler core::IScaler<double,3>
        builder.registerType<cpu_backend::ArrayScaler<double, 3>>()
            .as<core::IScaler<core::IArray<double, 3>>>()
            .singleInstance();

        //FlowField Scaler
        builder.registerType<cpu_backend::FlowFieldScaler>()
            .as< core::IScaler<core::IArray<double, 3>>>()
            .singleInstance();

        //GrayPenaltyCrossProblem Scaler
        builder.registerType<cpu_backend::GrayPenaltyCrossProblemScaler>()
            .as<core::IScaler<core::IGrayPenaltyCrossProblem>>()
            .singleInstance();

        //Problem Factory
        builder.registerType<cpu_backend::ProblemFactory>()
            .as<core::IProblemFactory>();

        //SunBakerLSUpdater
        builder.registerType<cpu_backend::SunBakerLSUpdater>()
            .as<optflow_solvers::ISunBakerLSUpdater>();

    }
    
    void SetCPUBackendDefaultSettings(Hypodermic::ContainerBuilder& builder)
    {
        auto penalty_settings = std::make_shared<cpu_backend::CharbonnierPenaltySettings>();
        builder.registerInstance<cpu_backend::CharbonnierPenaltySettings>(penalty_settings);

        auto median_filter_settings = std::make_shared< cpu_backend::CrossMedianFilterSettings>();
        builder.registerInstance<cpu_backend::CrossMedianFilterSettings>(median_filter_settings);

        auto ls_settings = std::make_shared< cpu_backend::LinearSystemSettings>();
        builder.registerInstance<cpu_backend::LinearSystemSettings>(ls_settings);
    }

    void SetCPUBackendCommandlineSettings(Hypodermic::ContainerBuilder& builder, boost::program_options::variables_map vm)
    {
        auto penalty_settings = std::make_shared<cpu_backend::CharbonnierPenaltySettings>();
        penalty_settings->DefaultBlendFactor = vm["penalty_blend"].as<double>();
        penalty_settings->Epsilon = vm["penalty_eps"].as<double>();
        penalty_settings->Exponent = vm["penalty_exp"].as<double>();
        builder.registerInstance<cpu_backend::CharbonnierPenaltySettings>(penalty_settings);

        auto median_filter_settings = std::make_shared< cpu_backend::CrossMedianFilterSettings>();
        median_filter_settings->FilterInfluence = vm["cross_filt"].as<double>();
        median_filter_settings->FilterLength = vm["cross_filt_len"].as<int>();
        median_filter_settings->SigmaColor = vm["cross_sig_col"].as<double>();
        median_filter_settings->SigmaDistance = vm["cross_sig_dist"].as<double>();
        median_filter_settings->SigmaDiv = vm["cross_sig_div"].as<double>();
        median_filter_settings->SigmaError = vm["cross_sig_err"].as<double>();
        builder.registerInstance<cpu_backend::CrossMedianFilterSettings>(median_filter_settings);

        auto ls_settings = std::make_shared< cpu_backend::LinearSystemSettings>();
        ls_settings->LambdaKernel = vm["lin_sys_lamb"].as<double>();
        builder.registerInstance<cpu_backend::LinearSystemSettings>(ls_settings);
    }
}