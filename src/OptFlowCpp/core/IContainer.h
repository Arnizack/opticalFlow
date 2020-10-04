#pragma once

namespace core
{

	template<class InnerTyp>
	class IContainer
	{
	public:
		virtual size_t Size() = 0;

		virtual bool CopyDataTo(InnerTyp*& destination) = 0;

	};

	
}