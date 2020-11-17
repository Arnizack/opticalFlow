#pragma once
#include<memory>

namespace core
{
	template<class T>
	using Scope = std::shared_ptr<T>;

	template<class T>
	using Ref = std::shared_ptr<T>;


}