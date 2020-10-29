#pragma once

namespace cpu_backend
{
	namespace _inner
	{
		//func(x,y)
		template<class CoordTyp,class FuncTyp>
		inline void Iterate2D(const CoordTyp& x_start,const CoordTyp& x_end,const CoordTyp& y_start,const CoordTyp& y_end, FuncTyp func)
		{
			
			for (CoordTyp y = y_start; y < y_end; y++)
			{
				for (CoordTyp x = x_start; x < x_end; x++)
				{
					func(x, y);
				}
			}
		}
	}
}