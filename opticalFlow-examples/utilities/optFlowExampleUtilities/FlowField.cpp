#include "FlowField.h"
#include<iostream>
#include<fstream>

bool utilities::loadFlowFromCSV(std::string path, FlowField& field)
{
	try {
		std::ifstream infile;
		infile.open(path);
		if (!infile.is_open())
			return false;
		char temp;
		infile >> field.width;
		infile >> temp;
		infile >> field.height;

		field.data.reserve(field.height * field.width * 2);

		//infile >> temp;

		float num;

		for (int counter = 0;
			infile >> num && counter < field.width * field.height * 2;
			counter++)
		{
			field.data.push_back(num);
			infile >> temp;
		}
	}catch (const std::exception & e) {
		return false;
	}
	return true;
	
}

bool utilities::saveFlowToCSV(std::string path, FlowField& field, char seperator)
{
	std::ofstream outfile;
	outfile.open(path, std::ios::binary | std::ios::out);

	if (!outfile.is_open())
		return false;

	outfile << field.width;
	outfile << seperator;
	outfile << field.height;

	outfile << std::endl;

	auto start = field.data.cbegin();
	auto end = field.data.cend();
	while (start!=end)
	{
		const auto& item = *start;
		outfile << item, sizeof(float);
		if(start!=end-1)
			outfile << seperator;
		start++;
	}

	outfile.close();

	return true;
}
