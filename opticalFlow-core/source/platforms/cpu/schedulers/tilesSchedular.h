#pragma once
#include<memory>
#include"platforms/cpu/datastructures/Vec2D.h"

namespace cpu
{

	template<typename... ARGS, typename _inst>
	inline void tilesSchedular2D(int Info, const Vec2D<int>& dimenions, const Vec2D<int>& tilesSize, const Vec2D<int>& padding,
		_inst Instruction, ARGS&&... args)
	{
		int2 idx;
		for (idx.y = -padding.y; idx.y < dimenions.y + padding.y; idx.y++)
		{
			for (idx.x = -padding.x; idx.x < dimenions.x + padding.x; idx.x++)
			{
			
				Instruction(idx, std::forward<ARGS>(args)...);
			}
		}

	}
	/*
	template<typename... ARGS, typename _inst>
	inline void tilesSchedular2DCF(int Info, const Vec2D<int>& dimenions, const Vec2D<int>& tilesSize, const Vec2D<int>& padding,
		_inst Instruction, ARGS&&... args)
	{
		Vec2D<int> idx;
		for (idx.x = -padding.x; idx.x < dimenions.x + padding.x; idx.x++)
		{
			for (idx.y = -padding.y; idx.y < dimenions.y + padding.y; idx.y++)
			{

				Instruction(idx, std::forward<ARGS>(args)...);
			}
		}

	}*/
}