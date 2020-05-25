#include"KDTreeMath.hpp"
#include<math.h>

double Erf(double x)
{
    // constants
    const double a1 = 0.254829592;
    const double a2 = -0.284496736;
    const double a3 = 1.421413741;
    const double a4 = -1.453152027;
    const double a5 = 1.061405429;
    const double p = 0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = abs(x);

    // A&S formula 7.1.26
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return sign * y;
}

float kdtree::CdfApprox(float x)
{
    const double InvSqrt2 =1/ sqrt(2);
	if (x < -2.5)
		return 0;
	if (x > 2.5)
		return 1;

	return static_cast<float>(0.5 * (1 + Erf(x * InvSqrt2)));
	
}

uint8_t kdtree::Floor(float x)
{
	return static_cast<uint8_t>(x);
    
}

float kdtree::DistanceSquared(KdTreeVal & a, KdTreeVal & b)
{
	float result = 0;
	for (int i = 0; i < 5; i++)
	{
		
		float difference = a[i] - b[i];
		result += difference * difference;
	}
	return result;
}
