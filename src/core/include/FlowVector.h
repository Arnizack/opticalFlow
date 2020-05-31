#include<stdint.h>
#pragma once
namespace core
{

struct FlowVector {
	int32_t vector_X;
	int32_t vector_Y;

	FlowVector(int32_t x, int32_t y);
	FlowVector();

	FlowVector operator*(float val) const;
	FlowVector operator/(float val) const;

	FlowVector operator+(int32_t val) const;
	FlowVector operator+(const FlowVector& val) const;
	FlowVector& operator+=(const FlowVector& val);

	FlowVector operator-(int32_t val) const;
	FlowVector operator-(const FlowVector& val) const;


};
}