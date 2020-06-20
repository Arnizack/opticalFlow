#include"FlowField.h"
#include<iostream>

int main()
{
	std::vector<float> expectedData =
	{ 1,1, 2,2, 3,3,
	4,4, 5,5, 6,6 };
	utilities::FlowField flowOut;
	flowOut.width = 2;
	flowOut.height = 3;
	flowOut.data = expectedData;

	utilities::saveFlowToCSV("testFlowField.csv", flowOut);

	utilities::FlowField flowIn;

	utilities::loadFlowFromCSV("testFlowField.csv", flowIn);

	if (flowIn.data == flowOut.data)
	{
		std::cout << "Das Laden und Speichern funktioniert" << std::endl;
		return 0;
	}
	else
	{
		std::cout << "Das Laden und Speichern funktioniert nicht !!!" << std::endl;
		return -1;
	}
	return 0;
	
}