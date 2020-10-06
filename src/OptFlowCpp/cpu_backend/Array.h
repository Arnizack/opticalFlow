#pragma once
#include "..\core\IArray.h"
#include <algorithm>
#include <vector>
#include<algorithm>

namespace cpu
{
	template<class InnerTyp, size_t DimCount>
	class Array : public core::IArray<InnerTyp, DimCount>
	{
	public:
		Array() = default;

		Array(std::array<const size_t, DimCount> shape, const size_t& size, const InnerTyp *const src) : core::IArray<InnerTyp, DimCount>(shape), _size(size), _data(std::vector<InnerTyp>(size))
		{
			std::copy_n(src, _size, _data.data());
		}

		Array(std::array<const size_t, DimCount> shape, const InnerTyp* const src) : core::IArray<InnerTyp, DimCount>(shape), _size(1), _data(std::vector<InnerTyp>(1))
		{
			for (size_t i = 0; i < DimCount; i++)
			{
				_size *= shape[i];
			}

			_data = std::vector<InnerTyp>(_size);
			std::copy_n(src, _size, _data.data());
		}

		Array(std::array<const size_t, DimCount> shape, const size_t& size, const InnerTyp& single_value) : core::IArray<InnerTyp, DimCount>(shape), _size(size), _data(std::vector<InnerTyp>(size))
		{
			for (size_t i = 0; i < size; i++)
			{
				_data[i] = single_value;
			}
		}

		Array(const size_t& size, const InnerTyp* const src) : _size(size), _data(std::vector<InnerTyp>(size)), Shape(std::array<const size_t, DimCount>())
		{
			Shape[0] = _size;
			for (size_t i = 1; i < DimCount; i++)
			{
				Shape[i] = 1;
			}

			std::copy_n(src, _size, _data.data());
		}

		inline InnerTyp& operator[](const size_t i)
		{
			return _data[i];
		}

		inline const InnerTyp& operator[](const size_t i) const
		{
			return _data[i];
		}

		virtual size_t Size() override
		{
			return _size;
		}

		virtual bool CopyDataTo(InnerTyp* destination) override
		{
			destination = InnerTyp[_size];
			std::copy_n(_data.data(), _size, destination);
			return true;
		}
	protected:
	private:
		size_t _size;
		std::vector<InnerTyp> _data;
	};
}