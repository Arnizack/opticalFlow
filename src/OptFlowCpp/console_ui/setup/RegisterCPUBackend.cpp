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
#include"cpu_backend/Scaler.h"

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
       ////Pyramid Builder
       //builder.registerType<cpu_backend::PyramidBuilder<std::shared_ptr<core::IGrayPenaltyCrossProblem>>>()
       //    .as< core::IPyramidBuilder< std::shared_ptr<core::IGrayPenaltyCrossProblem>>>();

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
}