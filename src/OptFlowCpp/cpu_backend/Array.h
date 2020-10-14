#pragma once
#include "core\IArray.h"
#include <algorithm>
#include <vector>

namespace cpu_backend
{
	template<class InnerTyp, size_t DimCount>
	class Array : public core::IArray<InnerTyp, DimCount>
	{
	public:
		Array() = default;

		Array(const std::array<const size_t, DimCount>& shape, const InnerTyp *const src) : 
			core::IArray<InnerTyp, DimCount>(shape), _size(1), _data(std::vector<InnerTyp>())
		{

			for (const size_t& dim : shape)
			{
				_size *= dim;
			}
			_data = std::vector<InnerTyp>(_size);

			std::copy_n(src, _size, _data.data());
		}

		Array(const std::array<const size_t, DimCount>& shape, core::IContainer<InnerTyp>& copy) :
			core::IArray<InnerTyp, DimCount>(shape), _size(copy.Size()), _data(std::vector<InnerTyp>(copy.Size()))
		{
			copy.CopyDataTo(_data.data());
		}

		Array(const std::array<const size_t, DimCount>& shape, const InnerTyp& single_value) :
			core::IArray<InnerTyp, DimCount>(shape), _size(1), _data(std::vector<InnerTyp>())
		{

			for (const size_t& dim : shape)
			{
				_size *= dim;
			}
			_data = std::vector<InnerTyp>(_size);
			for (size_t i = 0; i < _size; i++)
			{
				_data[i] = single_value;
			}

		}

		Array(const std::array<const size_t, DimCount>& shape) : 
			core::IArray<InnerTyp, DimCount>(shape), _size(1), _data(std::vector<InnerTyp>())
		{
			for (const size_t& dim : shape)
			{
				_size *= dim;
			}

			_data = std::vector<InnerTyp>(_size);
		}

		inline InnerTyp& operator[](const size_t i)
		{
			return _data[i];
		}

		inline const InnerTyp& operator[](const size_t i) const
		{
			return _data[i];
		}

		inline InnerTyp& operator[](const std::array<const size_t, DimCount>& coord)
		{
			//{y,x}
			size_t flattened_coord = 0;
			size_t multiplyer = 1;
			for (int i = DimCount - 1; i >= 0 ; i--)
			{
				flattened_coord += multiplyer * coord[i];
				multiplyer *= this->Shape[i];
			}
			return _data[flattened_coord];
		}

		virtual size_t Size() const override
		{
			return _size;
		}

		virtual bool CopyDataTo(InnerTyp* destination) override
		{
			std::copy_n(_data.data(), _size, destination);
			return true;
		}

		InnerTyp* Data()
		{
			return _data.data();
		}

		

	protected:
	private:
		size_t _size;
		std::vector<InnerTyp> _data;
	};
}