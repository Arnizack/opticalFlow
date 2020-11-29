#pragma once
#include<memory>
#include"commands/ICLOption.h"
#include"commands/ICLArgument.h"
#include<vector>

namespace console_ui
{
	class ICLEngine
	{
	public:
		virtual void PrintHelp() = 0;
		virtual void RegisterStartCommand(std::shared_ptr<ICLCommand> command) = 0;
		virtual void RegisterEndCommand(std::shared_ptr<ICLCommand> command) = 0;
		virtual void RegisterOption(std::shared_ptr<ICLOption> option) = 0;
		virtual void RegisterArgument(std::shared_ptr<ICLArgument> argument, int position) = 0;
		virtual void Parse(std::vector<std::string>& arguments) = 0;
	};
}