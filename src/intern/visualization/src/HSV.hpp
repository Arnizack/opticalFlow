#pragma once
namespace visualization
{
	struct HSV
	{
		//H is between 0 360
		float H;
		//S is between 0 1
		float S;
		//V is between 0 1
		float V;

		HSV(float h, float s, float v);
		HSV();
	}; 

}
                