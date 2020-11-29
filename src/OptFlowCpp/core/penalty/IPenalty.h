#pragma once
namespace core
{

	template<class T>
	class IPenalty
	{
	public:
		virtual T ValueAt(const T& x) = 0;

		virtual T FirstDerivativeAt(const T& x) = 0;

		virtual T SecondDerivativeAt(const T& x) = 0;
	};

}