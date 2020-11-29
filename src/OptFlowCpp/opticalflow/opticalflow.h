#include<memory>
#include<vector>
#include<string>
#include"optflow_composition/ContainerOptions.h"
namespace opticalflow
{
    struct Image
    {
        size_t width;
        size_t height;
        size_t color_bands;
        std::shared_ptr<std::vector<float>> data;
    };

    struct FlowField
    {
        size_t width;
        size_t height;
        std::shared_ptr<std::vector<double>> data;
    };

    using SolverOptions = optflow_composition::ContainerOptions;    

    void SaveFlowField(std::string filepath, FlowField flow);
    void SaveFlowFieldToColor(std::string filepath, FlowField flow);

    Image OpenImage(std::string filepath);

    std::shared_ptr<SolverOptions> ReadOptions(std::string filepath);
    
    class OpticalFlowSolver
    {
    public:
        virtual FlowField Solve(Image first_frame, Image second_frame) = 0;
    };

    std::shared_ptr<OpticalFlowSolver> CreateSolver(std::shared_ptr<SolverOptions> options = std::make_shared<SolverOptions>());

}