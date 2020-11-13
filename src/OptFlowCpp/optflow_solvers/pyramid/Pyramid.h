#pragma once
#include <memory>
#include <vector>
#include <array>
#include "core/pyramid/IPyramid.h"
#include "core/solver/problem/IProblemFactory.h"
#include "core/IScaler.h"

namespace cpu_backend
{
	template<class T>
	class Pyramid : public core::IPyramid<T>
	{
	public:
		Pyramid(std::vector<std::shared_ptr<T>> levels)
			: _levels(levels), iter(levels.size() - 1)
		{}

		virtual std::shared_ptr<T> NextLevel() override
		{
			return _levels[--iter];
		}

		virtual bool IsEndLevel() override
		{
			if (iter == 0)
				return true;
			return false;
		}

	private:
		size_t iter;
		std::vector<std::shared_ptr<T>> _levels; // MAX to min
	};
}