#pragma once
#include"ICLCommand.h"
namespace console_ui
{
	class ICLArgument : public ICLCommand
	{
	public:
		const std::string Name;
	};
}