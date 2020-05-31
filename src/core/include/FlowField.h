#pragma once
#include<array>
#include<stdint.h>
#include<fstream>
#include<iostream>
#include"FlowVector.h"

namespace core
{
	

	class FlowField
	{
	public:
		FlowField(uint32_t width, uint32_t height);
		//~FlowField();
		FlowVector GetVector(uint32_t x, uint32_t y) const;
		void SetVector(uint32_t x, uint32_t y, FlowVector vector);
		uint32_t GetWidth() const;
		uint32_t GetHeight() const;
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