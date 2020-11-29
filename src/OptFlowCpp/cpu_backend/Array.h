#pragma once
#include "core\IArray.h"
#include "Container.h"
#include <algorithm>
#include <memory>
#include <vector>

namespace cpu_backend
{
	template<class InnerTyp, size_t DimCount>
	class Array : public core::IArray<InnerTyp, DimCount>, public Container<InnerTyp>
	{
	public:

		/*
		* Constructor
		*/
		Array() = default;

		Array(const std::array<const size_t, DimCount>& shape, const InnerTyp *const src) : 
			core::IArray<InnerTyp, DimCount>(shape), _data( std::make_shared<std::vector<InnerTyp>>() )
		{
			size_t size = 1;
			
			for (const size_t& dim : shape)
			{
				size *= dim;
			}

			_data = std::make_shared<std::vector<InnerTyp>>(std::vector<InnerTyp>(size));

			std::copy_n(src, size, _data->data());
		}

		Array(const std::array<const size_t, DimCount>& shape, core::IContainer<InnerTyp>& copy) :
			core::IArray<InnerTyp, DimCount>(shape), _data( std::make_shared<std::vector<InnerTyp>>(std::vector<InnerTyp>(copy.Size())) )
		{
			copy.CopyDataTo(_data->data());
		}

		Array(const std::array<const size_t, DimCount>& shape, Container<InnerTyp>& copy) :
			core::IArray<InnerTyp, DimCount>(shape), _data(/*std::make_shared<std::vector<InnerTyp>>(std::vector<InnerTyp>(copy.Size()))*/)
		{
			_data = copy.GetRef();
		}

		Array(const std::array<const size_t, DimCount>& shape, const InnerTyp& single_value) :
			core::IArray<InnerTyp, DimCount>(shape), _data( std::make_shared<std::vector<InnerTyp>>() )
		{
			size_t size = 1;

			for (const size_t& dim : shape)
			{
				size *= dim;
			}

			_data = std::make_shared<std::vector<InnerTyp>>(std::vector<InnerTyp>(size));

			for (size_t i = 0; i < size; i++)
			{
				(*_data)[i] = single_value;
			}

		}

		Array(core::IArray<InnerTyp, DimCount>& copy) :
			core::IArray<InnerTyp, DimCount>(copy.Shape), _data( std::make_shared<std::vector<InnerTyp>>(std::vector<InnerTyp>(copy.Size())) )
		{
			copy.CopyDataTo(_data->data());
		}

		Array(const std::array<const size_t, DimCount>& shape) : 
			core::IArray<InnerTyp, DimCount>(shape), _data( std::make_shared<std::vector<InnerTyp>>() )
		{
			size_t size = 1;

			for (const size_t& dim : shape)
			{
				size *= dim;
			}

			_data = std::make_shared<std::vector<InnerTyp>>(std::vector<InnerTyp>(size));
		}

		/*
		* INTERFACE
		*/
		virtual size_t Size() const override
		{
			return _data->size();
		}

		virtual bool CopyDataTo(InnerTyp* destination) override
		{
			std::copy_n(_data->data(), _data->size(), destination);
			return true;
		}

		
		/*
		* CpuContainer
		*/
		virtual InnerTyp* Data() override
		{
			return _data->data();
		}

		virtual std::shared_ptr<std::vector<InnerTyp>> GetRef() override
		{
			return _data;
		}

		/*
		* CUSTOM
		*/
		inline InnerTyp& operator[](const size_t i)
		{
			return (*_data)[i];
		}

		inline const InnerTyp& operator[](const size_t i) const
		{
			return (*_data)[i]/*[i]*/;
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
			return (*_data)[flattened_coord];
		}

		

	protected:
	private:
		std::shared_ptr<std::vector<InnerTyp>> _data;
	};
}