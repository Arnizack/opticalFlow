#include<logger.hpp>
#pragma once
#include<vector>
#include<sstream>

#include <type_traits>
#include <utility>
#include <iostream>


namespace logger
{
	void logLine(int logLevel, char zeichen, int length);
	void logLine(int logLevel, char zeichen);
	void logLine(int logLevel, int length);
	void logLine(int logLevel);
	void logLine();


	template<typename... argsTypes>
	void logArgsDefault(const argsTypes & ...args)
	{
		logger::logArgs(10, 20, args...);
	}

	namespace internal
	{

		template<typename S, typename T>
		class is_streamable
		{
			template<typename SS, typename TT>
			static auto test(int)
				-> decltype(std::declval<SS&>() << std::declval<TT>(), std::true_type());

			template<typename, typename>
			static auto test(...)->std::false_type;

		public:
			static const bool value = decltype(test<S, T>(0))::value;
		};

		template<typename T,
			typename std::enable_if<is_streamable<std::ostringstream,T>::value, std::nullptr_t>::type = nullptr >
		std::string toString(const T& t)
		{
			std::ostringstream ss;
			ss << t;
			return ss.str();
		}
		
		template<typename T,
			typename std::enable_if<!is_streamable<std::ostringstream, T>::value, std::nullptr_t>::type = nullptr >
			std::string toString(const T& t)
		{
			std::ostringstream ss;
			ss << &t;
			return ss.str();
		}

		
		//
		template<typename T, typename... argsTypes>
		void __iterateArg(std::shared_ptr<std::ostringstream> fullStream, const size_t maxLength, const T& arg1, const argsTypes&... args)

		{


			std::string strArg = toString(arg1);

			for (size_t i = 0; i < maxLength; i++)
			{
				if (i < strArg.size())
				{
					fullStream->put(strArg.at(i));
				}
				else
				{
					fullStream->put(' ');
				}
			}

			fullStream->put('|');


			__iterateArg(fullStream, maxLength, args...);


		}

		inline void __iterateArg(std::shared_ptr<std::ostringstream> fullStream, const size_t maxLength)
		{

		}
		

	}

	template<typename... argsTypes>
	void logArgs(int level, const size_t maxLength,const argsTypes&... args)
	{


		std::shared_ptr<std::ostringstream>prtss(new std::ostringstream());
		prtss->clear();
		constexpr size_t argsLength = sizeof...(argsTypes);


		if (argsLength > 0)
		{
			internal::__iterateArg(prtss, maxLength, args...);
		}
		
		
		logger::log(level, prtss->str());
	}

}