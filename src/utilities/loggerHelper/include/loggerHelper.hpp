#include<logger.hpp>
#pragma once
#include<vector>
#include<array>
#include<sstream>

#include <type_traits>
#include <utility>
#include <iostream>
#include<iterator>

#define DEBUG_LEVEL 10
#define INFO_LEVEL 20
#define WARNING_LEVEL 30
#define ERROR_LEVEL 40
#define CRITICAL_LEVEL 50

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
		logger::logArgs(DEBUG_LEVEL, 20, args...);
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

		// To allow ADL with custom begin/end
		using std::begin;
		using std::end;



		template<typename T,
			typename std::enable_if<is_streamable<std::ostringstream,T>::value, std::nullptr_t>::type = nullptr >
		std::string toString(const T& t)
		{
			std::ostringstream ss;
			ss << t;
			return ss.str();
		}

		template<typename T>
		std::string toStringIterator(const T& vec)
		{
			std::ostringstream ss;
			ss << "[";
			std::string appender = "";
			for (const auto& item : vec)
			{
				ss << appender;
				ss << toString(item);
				appender = ", ";

			}
			ss << "]";

			return ss.str();
		}

		template<typename T>
		std::string toString(const std::vector<T>& vec)
		{
			return toStringIterator(vec);
		}

		template<typename T,std::size_t N>
		std::string toString(const std::array<T,N>& vec)
		{
			return toStringIterator(vec);
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

	template<typename T>
	void log2DData(int level, std::vector<std::vector<T>> data)
	{
		size_t maxSize = 0;

		std::vector<std::vector<std::string>> stringsVec;

		for (const auto& row : data)
		{
			std::vector<std::string> rowStr;
			for (const T& item : row)
			{
				auto dataStr = internal::toString(item);
				rowStr.push_back(dataStr);

				if (dataStr.size() > maxSize)
				{
					maxSize = dataStr.size();
				}

			}
			stringsVec.push_back(rowStr);
		}

		size_t stepSize = maxSize + 1;

		size_t idx;

		size_t counter;

		for (const auto& row : stringsVec)
		{
			std::string msg(stepSize*row.size(), ' ');
			
			idx = 0;
			for (const auto& strData : row)
			{
				counter = idx;
				for (const auto& c : strData)
				{
					msg[counter] = c;
					counter++;
				}
				idx += stepSize;
			}


			logger::log(level, msg);

		}
	}

	template<typename T>
	void log2DData(std::vector<std::vector<T>> data)
	{
		log2DData(DEBUG_LEVEL, data);
	}

	void logFunctionBegin(int level, std::string functionName);

	void logFunctionBegin(std::string functionName);

	template<typename T>
	void logFunctionEnd(int level, std::string functionName, T returnData)
	{
		logger::log(level,"End of " + functionName + ": " + internal::toString(returnData));
	}

	template<typename T>
	void logFunctionEnd(std::string functionName, T returnData)
	{
		logFunctionEnd(DEBUG_LEVEL, functionName, returnData);
	}

}