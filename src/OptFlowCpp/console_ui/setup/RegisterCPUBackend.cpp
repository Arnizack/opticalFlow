#pragma once
#include"RegisterCPUBackend.h"
#include"cpu_backend/penalty/CharbonnierPenalty.h"
#include"cpu_backend/flow/CrossBilateralMedianFilter.h"
#include"cpu_backend/image/warper/GrayWarper.h"
#include"cpu_backend/image/inner/DerivativeCalculator.h"
#include"cpu_backend/linalg/ArithmeticBasic.h"
#include"cpu_backend/linalg/ArithmeticChained.h"
#include"cpu_backend/linalg/ArithmeticVector.h"
#include"cpu_backend/ArrayFactory.h"
#include"cpu_backend/pyramid/Pyramid.h"
#include"cpu_backend/pyramid/PyramidBuilder.h"
#include"cpu_backend/problem/ProblemFactory.h"

namespace console_ui
{
    void RegisterCPUBackend(Hypodermic::ContainerBuilder& builder)
    {
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

        //Array Factory
        builder.registerType<cpu_backend::ArrayFactory<double, 1>>()
            .as<core::IArrayFactory<double, 1>>()
            .singleInstance();
        builder.registerType<cpu_backend::ArrayFactory<double, 3>>()
            .as<core::IArrayFactory<double, 3>>()
            .singleInstance();

        //Pyramid Builder
        builder.registerType<cpu_backend::PyramidBuilder<std::shared_ptr<core::IGrayPenaltyCrossProblem>>>()
            .as< core::IPyramidBuilder< std::shared_ptr<core::IGrayPenaltyCrossProblem>>>();

        //Problem Factory
        builder.registerType<core::IPenaltyProblem>()
            .as<core::IProblemFactory>();

    }
    void _RegisterCPULinalg(Hypodermic::ContainerBuilder& builder)
    {
        builder.registerType<cpu_backend::ArithmeticChained<double,1>>()
            .as<core::IArithmeticChained<double,1>>()
            .as<core::IArithmeticBasic<double,1>>()
            .singleInstance();
        builder.registerType<cpu_backend::ArithmeticVector<double, 1>>()
            .as<core::IArithmeticVector<double, 1>>()
            .singleInstance();

    }
    void SetCPUBackendDefaultSettings(Hypodermic::ContainerBuilder& builder)
    {
        auto penalty_settings = std::make_shared<cpu_backend::CharbonnierPenaltySettings>();
        builder.registerInstance<cpu_backend::CharbonnierPenaltySettings>(penalty_settings);
    }
}