#pragma once
#include "core/image/IPreProcessor.h"

#include "cpu_backend/ArrayFactory.h"
#include "cpu_backend/Statistics.h"
#include "cpu_backend/linalg/ArithmeticChained.h"
#include "cpu_backend/linalg/ArithmeticVector.h"
#include "cpu_backend/image/inner/convolution2D.h"
#include "cpu_backend/image/inner/convolution1D.h"

#include <memory>


namespace cpu_backend
{
    struct ROFPreProcessingSettings
    {
        size_t iter = 100;
        float lambda = 0.125;
        float tau = 0.25;
        float sub_factor = 0.95;
    };


    class ROFPreProcessing : public core::IPreProcessor<float, 3>
    {

        using PtrImage = std::shared_ptr<core::IArray<float, 3>>;

    public:
        ROFPreProcessing(std::shared_ptr<ROFPreProcessingSettings> settings,
            std::shared_ptr<core::IArrayFactory<float, 2>> arr_factory2D, std::shared_ptr<core::IArrayFactory<float, 3>> arr_factory3D, 
            std::shared_ptr<core::IStatistics<float>> statistic, std::shared_ptr<core::IArithmeticVector<float, 2>> arith_vector);


        virtual PtrImage Process(PtrImage img) override;
        

        void pi_gradient_descend(float* img, const float& img_var, const int& width, const int& height, float* destination);

    private:
        void calculate_next_gradients(float* current_x_gradient, float* current_y_gradient, float* img,
            const int& width, const int& height, float* destination_x_gradient, float* destination_y_gradient);

        void divergence_func(float* x_gradient, float* y_gradient, const int& width, const int& height, float* dst);

        void delta_func(float* image, const int& width, const int& height, float* dst_X_gradient, float* dst_Y_gradient);

        const size_t _iter;
        float _lambda;
        const float _tau;
        const float _sub_factor;

        std::shared_ptr<core::IArrayFactory<float, 2>> _arr_factory2D;
        std::shared_ptr<core::IArrayFactory<float, 3>> _arr_factory3D;
        std::shared_ptr<core::IStatistics<float>> _statistic;
        std::shared_ptr<core::IArithmeticVector<float, 2>> _arith_vector;
    };
}