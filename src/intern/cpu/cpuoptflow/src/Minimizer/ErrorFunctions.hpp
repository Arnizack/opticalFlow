#include<math.h>

float CharbonnierError(float xSquart)
{
	const float epsilon = 0.001;
	const float epsilonSquart = epsilon * epsilon;

	return sqrt(xSquart + epsilonSquart);
}