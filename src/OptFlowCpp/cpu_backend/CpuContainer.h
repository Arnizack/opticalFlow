#pragma once
#include "..\core\IContainer.h"
#include "eigen3/Eigen/Dense"
#include <memory>
#include <initializer_list>

namespace cpu
{
	template<class InnerTyp>
	class Container : public core::IContainer<InnerTyp>
	{
	public:
		Container() = default;
		Container(const size_t& size) : _size(size), _data(Eigen::Array<InnerTyp, Eigen::Dynamic, 1>(size, 1)) {}
		Container(const size_t& size, const InnerTyp* const src) : _size(size), _data(Eigen::Array < InnerTyp, Eigen::Dynamic, 1 > (size, 1))
		{
			for (int i = 0; i < _size; i++)
			{
				_data[i] = src[i];
			}
		}
		Container(std::initializer_list<InnerTyp> args) : _size(args.size()), _data(Eigen::Array<InnerTyp, Eigen::Dynamic, 1>(args.size(), 1)) 
		{
			int i = 0;
			const InnerTyp* current = args.begin();
			while (current != args.end())
			{
				_data[i] = *current;
				++current;
				++i;
			}
		}

		inline InnerTyp& operator[](const size_t& i)
		{
			return _data[i];
		}
		inline const InnerTyp& operator[](const size_t& i) const
		{
			return _data[i];
		}

		virtual size_t Size() override
		{
			return _size;
		}

		virtual bool CopyDataTo(InnerTyp*& destination) override
		{
			destination = &_data[0];
			return true;
		}

	protected:
	private:
		const size_t _size;
		Eigen::Array<InnerTyp, Eigen::Dynamic, 1> _data;
	};
}