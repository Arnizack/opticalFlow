#pragma once
#include<logger.hpp>
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



	/*
	Example: logLine(5,"a",2)
	Logges at Level 5:
	"aa"
	*/
	void logLine(int logLevel, char symbol, int length);
	/*
	Example: logLine(5,"a")
	Logges at Level 5:
	"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	*/
	void logLine(int logLevel, char symbol);
	/*
	Example: logLine(5,2)
	Logges at Level 5:
	"--"
	*/
	void logLine(int logLevel, int length);

	/*
	Example: logLine(5)
	Logges at Level 5:
	"---------------------------------------------"
	*/
	void logLine(int logLevel);
	/*
	Example: logLine()
	Logges at Level 10:
	"---------------------------------------------"
	*/
	void logLine();


	/* Example: logArgsDefault(4,0.1,std::vector<int>{1,2,3},"test")
	logges at level 10:
	4	|0.1	|[1,2,3]	|test
	*/
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

	/* Example: logArgs(42,4,0.1,std::vector<int>{1,2,3},"test")
	logges at level 42:
	4	|0.1	|[1,2,3]	|test
	*/
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

	/*
	log2DData logges a 2D vector as a table 
	*/
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



	/*
	log2DData logges a 2D vector as a table
	with default level 10
	*/
	template<typename T>
	void log2DData(std::vector<std::vector<T>> data)
	{
		log2DData(DEBUG_LEVEL, data);
	}

	/*
	Example: logFunctionBegin(42,"test Func")
	logges at Level 42:
	-------------------------------------
	test func
	-------------------------------------
	*/
	void logFunctionBegin(int level, std::string functionName);

	/*
	Example: logFunctionBegin("test Func")
	logges at Level 10:
	-------------------------------------
	test func
	-------------------------------------
	*/
	void logFunctionBegin(std::string functionName);

	/*
	Example: logFunctionEnd(42,"test func",2)
	logges at level 42:
	End of test func: 2
	*/
	template<typename T>
	void logFunctionEnd(int level, std::string functionName, T returnData)
	{
		logger::log(level,"End of " + functionName + ": " + internal::toString(returnData));
	}

	/*
	Example: logFunctionEnd("test func",2)
	logges at level 10:
	End of test func: 2
	*/
	template<typename T>
	void logFunctionEnd(std::string functionName, T returnData)
	{
		logFunctionEnd(DEBUG_LEVEL, functionName, returnData);
	}

}