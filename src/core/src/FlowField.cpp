#include "FlowField.h"



core::FlowField::FlowField(uint32_t width, uint32_t height)
{
}

core::FlowField::~FlowField()
{
}

core::FlowVector core::FlowField::GetVector(uint32_t x, uint32_t y)
{
	return core::FlowVector();
}

void core::FlowField::SetVector(uint32_t x, uint32_t y, FlowVector)
{
}

uint32_t core::FlowField::GetWidth() const
{
	return uint32_t();
}

uint32_t core::FlowField::GetHeight()
{
	return uint32_t();
}

const core::FlowVector * core::FlowField::Data()
{
	return nullptr;
}

void core::FlowField::Save(std::string filepath)
{
}

core::FlowField core::FlowField::Upsize(uint32_t target_wight, uint32_t target_height)
{
	
	return *this;
}
