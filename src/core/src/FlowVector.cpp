#include"FlowVector.h"

namespace core
{
	FlowVector FlowVector::operator*(float val) const
	{
		FlowVector result;
		result.vector_X = vector_X * val;

		result.vector_Y = vector_Y * val;
		return result;
	}

	FlowVector FlowVector::operator/(float val) const
	{
		FlowVector result;
		result.vector_X = vector_X / val;

		result.vector_Y = vector_Y / val;
		return result;
	}

	FlowVector FlowVector::operator+(int32_t val) const
	{
		FlowVector result;
		result.vector_X = vector_X + val;

		result.vector_Y = vector_Y + val;
		return result;
	}
	FlowVector FlowVector::operator+(const FlowVector& val) const
	{
		FlowVector result;
		result.vector_X = vector_X + val.vector_X;

		result.vector_Y = vector_Y + val.vector_Y;
		return result;
	}
	FlowVector FlowVector::operator-(int32_t val) const
	{
		FlowVector result;
		result.vector_X = vector_X - val;

		result.vector_Y = vector_Y - val;
		return result;
	}
	FlowVector FlowVector::operator-(const FlowVector& val) const
	{
		FlowVector result;
		result.vector_X = vector_X - val.vector_X;

		result.vector_Y = vector_Y - val.vector_Y;
		return result;
	}
	FlowVector::FlowVector(int32_t x, int32_t y)
	{
		vector_X = x;

		vector_Y = y;
	}

	FlowVector::FlowVector()
	{
		vector_X = 0;

		vector_Y = 0;
	}
	FlowVector& FlowVector::operator+=(const FlowVector& val) {
		vector_X += val.vector_X;
		vector_Y += val.vector_Y;
		return *this;
	}
}