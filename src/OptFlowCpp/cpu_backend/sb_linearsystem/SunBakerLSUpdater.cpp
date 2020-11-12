#pragma once
#include"pch.h"
#include"SunBakerLSUpdater.h"
#include"SunBakerLinearOp.h"

namespace cpu_backend
{
    using PtrGrayImg = std::shared_ptr<core::IArray<float, 2>>;
    using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
    void SunBakerLSUpdater::SetFramePair(PtrGrayImg first_image, PtrGrayImg second_image)
    {
        size_t img_width = first_image->Shape[1];
        size_t img_height = first_image->Shape[0];

        size_t img_size = img_width * img_height;
        
        std::array<const size_t, 1> shape = { img_size };

        initialize_deriv(img_size);
        _width = img_width;
        _height = img_height;

        float* I_x = _image_deriv_x->Data();
        float* I_y = _image_deriv_y->Data();
        float* I_t = _image_deriv_time->Data();

        float* first_image_data = std::static_pointer_cast<
            Array<float, 2>>(first_image)->Data();
        float* second_image_data = std::static_pointer_cast<
            Array<float, 2>>(second_image)->Data();

        
        //error Shape
        auto mixed_img = std::make_unique<Array<float, 1>>(shape);
        float* ptr_mixed_img = mixed_img->Data();

        float b = 0.4;

        for (int i = 0; i < img_size; i++)
            ptr_mixed_img[i] = first_image_data[i] * b + second_image_data[i] * (1 - b);


        _deriv_calculator->ComputeDerivativeX(
            ptr_mixed_img,img_width,img_height,I_x);
        _deriv_calculator->ComputeDerivativeY(
            ptr_mixed_img, img_width, img_height, I_y);
        for (int i = 0; i < img_size; i++)
            I_t[i] = second_image_data[i] - first_image_data[i];

    }
    void SunBakerLSUpdater::UpdateParameter(PtrFlowField linearization_points, double relaxation)
    {
        size_t size = _image_deriv_x->Shape[0];
        initialize_data(size);
        double* D_y = _data_y->Data();
        double* D_x = _data_x->Data();
        double* R = _rest->Data();

        float* I_x = _image_deriv_x->Data();
        float* I_y = _image_deriv_y->Data();
        float* I_t = _image_deriv_time->Data();
        double* ptr_flow_y = std::static_pointer_cast<Array<double, 3>>(linearization_points)->Data();
        double* ptr_flow_x = ptr_flow_y + size;
        double* b_y = _desired_result->Data();
        double* b_x = b_y + size;

        for (int i = 0; i < size; i++)
        {
            /*
            du : flow_x
            dv : flow_y

            p'_D first derivative of the penalty function at I_t + I_x du + I_y dv
            p''_D second derivative of the penalty function at I_t + I_x du + I_y dv

            A = |K+D_y R    |
                |R     K+D_x|

            b = |-I_y p'_D + I_y p''_D I_y dv + I_x p''_D I_y du + lambda_R 2 v_relax|
                |-I_x p'_D + I_x p''_D I_x du + I_x p''_D I_y dv + lambda_R 2 u_relax|

            K = - 2 lambda_K * mat(ker(x))
            D_y = diag(I_y p''_D I_y + 2 lambda_R)
            R = diag(I_x p''_D I_y)
            D_x = diag(I_x p''_D I_x + 2 lambda_R )
            */
            double flow_x = ptr_flow_x[i];
            double flow_y = ptr_flow_y[i];
            double img_linearized = I_x[i] * flow_x + I_y[i] * flow_y + I_t[i];
            double penalty_first_deriv = _penalty->FirstDerivativeAt(img_linearized);
            double penalty_second_deriv = _penalty->SecondDerivativeAt(img_linearized);

            D_y[i] = I_y[i] * penalty_second_deriv * I_y[i] + 2 * relaxation;
            D_x[i] = I_x[i] * penalty_second_deriv * I_x[i] + 2 * relaxation;
            R[i] = I_x[i] * penalty_second_deriv * I_y[i];
            b_y[i] = -I_y[i] * penalty_first_deriv + I_y[i] * penalty_second_deriv * I_y[i] * flow_y;
            b_y[i] += R[i] * flow_x + relaxation * 2 * flow_y;
            b_x[i] = -I_x[i] * penalty_first_deriv + I_x[i] * penalty_second_deriv * I_x[i] * flow_x;
            b_x[i] += R[i] * flow_y + relaxation * 2 * flow_x;
        }
    }
    std::shared_ptr<core::ILinearProblem<double>> SunBakerLSUpdater::Update()
    {
        auto problem = std::make_shared<core::ILinearProblem<double>>();
        problem->LinearOperator = std::make_shared<SunBakerLinearOp>(_width,_height,_data_y, _data_x,_rest,_lambda_kernel);
        problem->Vector = _desired_result;
        return problem;
    }
    void SunBakerLSUpdater::SetPenalty(std::shared_ptr<core::IPenalty<double>> penalty)
    {
        _penalty = penalty;
    }
    SunBakerLSUpdater::SunBakerLSUpdater(
        std::shared_ptr<DerivativeCalculator<float>> deriv_calculator,
        std::shared_ptr< LinearSystemSettings> settings)
        :_deriv_calculator(deriv_calculator), _lambda_kernel(settings->LambdaKernel)
    {
    }
    void SunBakerLSUpdater::initialize_deriv(size_t size)
    {

        std::array<const size_t, 1> shape= { size };

        if (_image_deriv_x == nullptr || _image_deriv_x->Size() != size)
        {
            _image_deriv_x = std::make_shared<Array<float, 1>>(shape);
            _image_deriv_y = std::make_shared<Array<float, 1>>(shape);
            _image_deriv_time = std::make_shared<Array<float, 1>>(shape);

        }
        
    }
    void SunBakerLSUpdater::initialize_data(size_t data_size)
    {

        std::array<const size_t, 1> data_shape = { data_size };
        std::array<const size_t, 1> desired_shape = { data_size * 2 };

        if (_data_x == nullptr || _data_x->Size() != data_size)
        {
            _data_x = std::make_shared<Array<double, 1>>(data_shape);
            _data_y = std::make_shared<Array<double, 1>>(data_shape);
            _rest = std::make_shared<Array<double, 1>>(data_shape);
            _desired_result = std::make_shared<Array<double, 1>>(desired_shape);
        }
    }
}