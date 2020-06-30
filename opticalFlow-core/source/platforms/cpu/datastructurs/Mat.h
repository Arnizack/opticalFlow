#pragma once
#include"Vec2D.h"
#include<memory>
#include<numeric>
#include<vector>


template<typename T>
class TD;

namespace cpu {
	
	template<class InputIt, class OutputIt, class BinaryOperator>
	void _combinedAssignmentOperators(InputIt begin_in, InputIt end_in, OutputIt begin_out, BinaryOperator bin_operator)
	{
		while (begin_in != end_in)
		{
			bin_operator(*begin_in , *begin_out);
			begin_in++;
			begin_out++;
		}
	}

	template<class InputIt, class InputIt1, class InputIt2, class BinaryOperator>
	void _arithmeticOperators(InputIt1 begin_1in, InputIt1 end_1in, InputIt2 begin_2in, InputIt1 begin_out, BinaryOperator bin_operator)
	{
		while (begin_1in != end_1in)
		{
			*begin_out = bin_operator(*begin_1in, *begin_2in);
			begin_1in++;
			begin_2in++;
			begin_out++;
		}
	}
	
	template<typename _Ty>
	class Mat {

	public:

		

		Mat(const Vec2D<int> dimensions) :Dimensions(dimensions)
		{
			setupData();
			
		}

		Mat(const Vec2D<int> dimensions, const _Ty& defaultValue) :Dimensions(dimensions)
		{
			setupData();
			std::for_each(begin(), end(), [&](_Ty& val)
			{
				val = defaultValue;
			});
		}




		const Vec2D<int> Dimensions;

		constexpr auto begin() const
		{
			
			return data->begin();
		}

		constexpr auto end() const
		{
			return data->end();
		}

		template<typename T>
		Mat& operator+=(const Mat<T> other)
		{
			_Ty* it_begin_this = std::begin(*this);
			_Ty* it_end_this = std::end(*this);

			T* it_begin_other = std::begin(other);
			//auto* it_end_other = std::end(other);
			_combinedAssignmentOperators(it_begin_this, it_end_this, it_begin_other, [](_Ty& t, T& o)
			{t += o; }
			);

		}

		template<typename T>
		Mat& operator-=(const Mat<T> other)
		{
			_Ty* it_begin_this = std::begin(*this);
			_Ty* it_end_this = std::end(*this);

			T* it_begin_other = std::begin(other);
			//auto* it_end_other = std::end(other);
			_combinedAssignmentOperators(it_begin_this, it_end_this, it_begin_other, [](_Ty& t, T& o)
			{t -= o; }
			);

		}

		template<typename T>
		Mat& operator*=(const T& scalar)
		{
			std::for_each(begin(), end(), [](_Ty& cell)
			{
				cell*= scalar;
			});
		}

		template<typename T>
		Mat& operator/=(const T& scalar)
		{
			std::for_each(begin(), end(), [](_Ty& cell)
			{
				cell/= scalar;
			});
		}

		template<typename T>
		Mat operator+(const Mat<T> other) const
		{
			Mat<_Ty, _DimCount> output(Dimensions);
			_arithmeticOperators(std::begin(*this), std::end(*this), std::begin(other), std::begin(output),
			[](const _Ty& aa, const T& bb)
			{
				return aa + bb;
			}
			);
		}

		template<typename T>
		Mat operator-(const Mat<T> other) const
		{
			Mat<_Ty, _DimCount> output(Dimensions);
			_arithmeticOperators(std::begin(*this), std::end(*this), std::begin(other), std::begin(output),
				[](const _Ty& aa, const T& bb)
			{
				return aa - bb;
			}
			);
		}

		template<typename T>
		Mat operator*(const T& scalar) const
		{
			Mat<_Ty, _DimCount> output(Dimensions);
			auto beginIt_in = begin();
			auto endIt_in = end();
			auto beginIt_out = std::begin(output);
			
			while (beginIt_in!=beginIT_out)
			{
				*beginIt_out = (*beginIt_in) * scalar;
				*beginIt_in++;
				*beginIt_out++;
			}
		}

		template<typename T>
		Mat operator/(const T& scalar) const
		{
			Mat<_Ty, _DimCount> output(Dimensions);
			auto beginIt_in = begin();
			auto endIt_in = end();
			auto beginIt_out = std::begin(output);

			while (beginIt_in != beginIT_out)
			{
				*beginIt_out = (*beginIt_in) / scalar;
				*beginIt_in++;
				*beginIt_out++;
			}
		}

		

		_Ty& operator[](const Vec2D<int>& coordinate)
		{
			

			size_t mem_idx = 0;	
			mem_idx = coordinate.x * Dimensions.y+ coordinate.y;

			return data->operator[](mem_idx);
		}

		std::shared_ptr<std::vector<_Ty>> data;
	protected:
		

		size_t Mem_size;

		/*
		x			|y		|z
		y_dim*z_dim	|z_dim	|1
		*/

	private:
		void setupData()
		{
			Mem_size = Dimensions.x * Dimensions.y;

			data = std::make_shared<std::vector<_Ty>>(Mem_size);
		}
	};
}