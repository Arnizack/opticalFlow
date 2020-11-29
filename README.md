
# opticalFlow
Der Versuch einen Optical Flow Algorithmus zu implementieren.

## how run the binaries

``` opticalflow.exe -1 <first image path> -2 <second image path> -O <flow output path> -F <flow visualization output path> -J <json settings>```

An example for the json settings:

```json 
{

"cpu_settings": {
	"charbonnier_penalty_defaultblendfactor": 0.0,
	"charbonnier_penalty_epsilon": 0.001,
	"charbonnier_penalty_exponent": 0.45,
	"cross_median_filter_filterinfluence": 5.0,
	"cross_median_filter_filterlength": 9,
	"cross_median_filter_sigmacolor": 0.08,
	"cross_median_filter_sigmadistance": 4.0,
	"cross_median_filter_sigmadiv": 0.3,
	"cross_median_filter_sigmaerror": 20.0,
	"linear_system_lambdakernel": 0.3

},

"solver_settings": {

	"convex_settings": {
		"cg_solver_iterations_convex": 100,
		"cg_solver_tolerance_convex": 0.001,
		"incremental_solver_steps_convex": 3,
		"linearization_solver_endrelaxation_convex": 0.0003921568627450981,
		"linearization_solver_relaxationsteps_convex": 1.0,
		"linearization_solver_startrelaxation_convex": 3.921568627450981e-07,
		"pyramid_resolution_minresolutionx_convex": 32,
		"pyramid_resolution_minresolutiony_convex": 32,
		"pyramid_resolution_scalefactor_convex": 0.5
	},

	"gnc_penalty_gncsteps": 2,
	"non_convex_settings": {
		"cg_solver_iterations_non_convex": 30,
		"cg_solver_tolerance_non_convex": 0.001,
		"incremental_solver_steps_non_convex": 3,
		"linearization_solver_endrelaxation_non_convex": 0.0003921568627450981,
		"linearization_solver_relaxationsteps_non_convex": 5.0,
		"linearization_solver_startrelaxation_non_convex": 3.921568627450981e-07,
		"pyramid_resolution_minresolutionx_non_convex": 400,
		"pyramid_resolution_minresolutiony_non_convex": 400,
		"pyramid_resolution_scalefactor_non_convex": 0.8

	}

}

}
```

## vcpkg config
vcpkg install spdlog
vcpkg install libpng
vcpkg install nlohmann-json
vcpkg install spdlog

## References
- https://vision.middlebury.edu/flow/data/
- A Quantitative Analysis of Current Practices in Optical Flow Estimation and the Principles behind Them (Deqing Sun. Stefan Roth, Michael J. Black)

- Horn–Schunck Optical Flow with a Multi-Scale Strategy (Enric Meinhardt-Llopis, Javier Sanchez, Daniel Kondermann)

- Determining Optical Flow (Berthold K.P. Horn, Brian G. Rhunck )
