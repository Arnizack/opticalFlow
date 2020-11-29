#pragma once
#include<string>
#include<memory>
#include"libs/Hypodermic/Hypodermic.h"
namespace console_ui
{
	class ICLCommand
	{
		const std::string Description = "";
		const std::string Tooltip = "";

		virtual bool IsValide(std::string value) = 0;
		virtual bool ErrorMessage(std::string value) = 0;
		virtual void Execute(std::string value, std::shared_ptr<Hypodermic::Container> di_container) = 0;
	};
}