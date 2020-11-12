#pragma once
#include"RegisterCPULinalg.h"
#include"cpu_backend/linalg/ArithmeticBasic.h"
#include"cpu_backend/linalg/ArithmeticChained.h"
#include"cpu_backend/linalg/ArithmeticVector.h"
#include"cpu_backend/Reshaper.h"
#include"cpu_backend/image/warper/GrayWarper.h"


namespace console_ui
{
    void RegisterCPULinalg(Hypodermic::ContainerBuilder& builder)
    {
        builder.registerType<cpu_backend::ArithmeticChained<double, 1>>()
            .as<core::IArithmeticChained<double, 1>>()
            .as<core::IArithmeticBasic<double, 1>>()
            .singleInstance();
        builder.registerType<cpu_backend::ArithmeticVector<double, 1>>()
            .as<core::IArithmeticVector<double, 1>>()
            .singleInstance();

        builder.registerType<cpu_backend::ArithmeticBasic<double, 3>>()
            .as< core::IArithmeticBasic<double, 3>>();

        //core::IArithmeticBasic<double, 3>
        builder.registerType<cpu_backend::ArithmeticVector<double, 3>>()
            .as<core::IArithmeticVector<double, 3>>()
            .singleInstance();
        //Resharper
        builder.registerType<cpu_backend::Reshaper<double>>()
            .as<core::IReshaper<double>>()
            .singleInstance();

        //Warper
        builder.registerType<cpu_backend::GrayWarper>()
            .as<core::IGrayWarper>()
            .singleInstance();
        /*
        std::shared_ptr<core::IReshaper<double>> flow_reshaper,
			std::shared_ptr<core::IGrayWarper> warper,
        */

        //Array Factory
        builder.registerType<cpu_backend::ArrayFactory<double, 1>>()
            .as<core::IArrayFactory<double, 1>>()
            .singleInstance();

        builder.registerType<cpu_backend::ArrayFactory<double, 3>>()
            .as<core::IArrayFactory<double, 3>>()
            .singleInstance();

        builder.registerType<cpu_backend::ArrayFactory<float, 2>>()
            .as<core::IArrayFactory<float, 2>>()
            .asSelf()
            .singleInstance();

        builder.registerType<cpu_backend::ArrayFactory<float, 3>>()
            .as<core::IArrayFactory<float, 3>>()
            .asSelf()
            .singleInstance();
    }
}