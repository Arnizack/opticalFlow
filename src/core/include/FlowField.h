#pragma once
#include<array>
#include<stdint.h>
#include<fstream>
#include<iostream>

namespace core
{
	struct FlowVector {
		std::int32_t vector_X;
		std::int32_t vector_Y;
	};

	class FlowField
	{
	public:
		FlowField(uint32_t width, uint32_t height);
		//~FlowField();
		FlowVector GetVector(uint32_t x, uint32_t y);
		void SetVector(uint32_t x, uint32_t y, FlowVector vector);
		uint32_t GetWidth() const;
		uint32_t GetHeight();
		const FlowVector* Data();
		void Save(std::string filepath);
		void Load(std::string filepath);

		FlowField Upsize(uint32_t target_wight, uint32_t target_height);

	private:
		std::uint32_t width;
		std::uint32_t height;
		FlowVector *field;
	};
}