#include "FlowField.h"



core::FlowField::FlowField(uint32_t width, uint32_t height)
{
	this->width = width;
	this->height = height;
	this->field = (core::FlowVector*) std::malloc(width*height * sizeof(core::FlowVector));

}

/*
core::FlowField::~FlowField()
{
	free(this->field);
}*/

core::FlowVector core::FlowField::GetVector(uint32_t x, uint32_t y)
{
	return this->field[x + y * this->width];
}

void core::FlowField::SetVector(uint32_t x, uint32_t y, FlowVector vector)
{
	this->field[x + y * this->width] = vector;
}

uint32_t core::FlowField::GetWidth() const
{
	return this->width;
}

uint32_t core::FlowField::GetHeight()
{
	return this->height;
}

const core::FlowVector * core::FlowField::Data()
{
	return this->field;
}

void core::FlowField::Save(std::string filepath)
{
	std::ofstream outfile;
	outfile.open("test.bin", std::ios::binary | std::ios::out);

	outfile.write((char*) &this->width, sizeof(std::uint32_t));
	outfile.write((char*) &this->height, sizeof(std::uint32_t));

	for (std::uint32_t i = 0; i < this->width * this->height; i++)
	{
		outfile.write((char*)&this->field[i].vector_X, sizeof(std::int32_t));
		outfile.write((char*)&this->field[i].vector_Y, sizeof(std::int32_t));
	}

	outfile.close();
}

void core::FlowField::Load(std::string filepath)
{
	std::ifstream infile;
	infile.open("test.bin", std::ios::in | std::ios::binary);
	
	infile.read((char*) &this->width, sizeof(std::uint32_t));
	infile.read((char*) &this->height, sizeof(std::uint32_t));
	
	std::int32_t temp_value;
	core::FlowVector temp_vector;

	for (std::uint32_t i = 0; i < this->width * this->height; i++)
	{
		infile.read((char*) &temp_value, sizeof(std::int32_t));
		temp_vector.vector_X = temp_value;
		infile.read((char*)&temp_value, sizeof(std::int32_t));
		temp_vector.vector_Y = temp_value;

		this->field[i] = temp_vector;
	}

	if (!infile)
	{
		//Error
		infile.gcount();
		infile.clear();
	}
}

core::FlowField core::FlowField::Upsize(uint32_t target_widht, uint32_t target_height)
{

	/*
	*	Scales the Flow Field up to the new Resolution target_width*target_height
	*	this implementation uses bilinear intrerpolation
	*	has to be updated to joint biliteral filtering (GPU compatible)
	*	returns a pointer to the new Flow Field
	*/
	
	//the new upscaled Field
	core::FlowField new_field (target_height, target_widht);

	core::FlowVector a, b, c, d, temp;

	std::int32_t original_width = this->width;
	std::int32_t original_height = this->height;

	double x_ratio = (original_width - 1) / target_widht;
	double y_ratio = (original_height - 1) / target_widht;
	double x_spacing, y_spacing;
	int x, y, index;

	for (std::uint32_t i = 0; i < target_height; i++) //columns
	{
		for (std::uint32_t j = 0; j < target_widht; j++) //rows
		{
			x = (int)(x_ratio * j);
			y = (int)(y_ratio * i);
			index = y * this->width + x;

			x_spacing = (x_ratio * j) - x;
			y_spacing = (y_ratio * i) - y;
			
			a = this->field[index];
			b = this->field[index + 1];
			c = this->field[index + original_width];
			d = this->field[index + original_width + 1];

			//temp = A * (1 - x_spacing) * (1 - y_spacing) + B * x_spacing * (1 - y_spacing) + C * (1 - x_spacing) * y_spacing + D *( x_spacing * y_spacing)
			temp.vector_X = (std::int32_t) (a.vector_X*(1 - x_spacing) * (1 - y_spacing) + b.vector_X * x_spacing * (1 - y_spacing) + c.vector_X * (1 - x_spacing) * y_spacing + d.vector_X * x_spacing * y_spacing);
			temp.vector_Y = (std::int32_t) (a.vector_Y*(1 - x_spacing) * (1 - y_spacing) + b.vector_Y * x_spacing * (1 - y_spacing) + c.vector_Y * (1 - x_spacing) * y_spacing + d.vector_Y * x_spacing * y_spacing);

			new_field.SetVector(j, i, temp);
		}
	}

	return new_field;
}
