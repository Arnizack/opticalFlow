#pragma once
#include<chrono>
#include<string>
#include<thread>
#include<mutex>
#include<fstream>
#include<iomanip>
#include<sstream>
#include <thread>
#include"core/Logger.h"

//Inspired by cherno's visual instrumentation
namespace debug_helper
{
    using FloatingPointMicroseconds = std::chrono::duration<double, std::micro>;

    struct ProfileResult
	{
		std::string Name;

		FloatingPointMicroseconds Start;
		std::chrono::microseconds ElapsedTime;
		std::thread::id ThreadID;
	};
    
    struct ProfileWriterSeason
    {
        std::string name;
    };

    class ProfileWriter
    {
    private:
		std::mutex _mutex;
		ProfileWriterSeason* _current_session;
		std::ofstream _output_stream;
    public:
        //can't be copied
		ProfileWriter(const ProfileWriter&) = delete;
		ProfileWriter(ProfileWriter&&) = delete;

		void BeginSession(const std::string& name, const std::string& filepath = "profiling_results.json")
		{
			std::lock_guard<std::mutex>  lock(_mutex);
			if (_current_session != nullptr)
			{
                OPF_LOG_ERROR("Profiler::BeginSession('{0}') when session '{1}' already open.", name, _current_session->name);
				InternalEndSession();
			}
			_output_stream.open(filepath);

			if (_output_stream.is_open())
			{
				_current_session = new ProfileWriterSeason({name});
				WriteHeader();
			}
			else
			{
				OPF_LOG_ERROR("Profiler::BeginSession('{0}') when session '{1}' already open.", name, _current_session->name);
			}
		}

		void EndSession()
		{
			std::lock_guard<std::mutex>  lock(_mutex);
			InternalEndSession();
		}

		void Write(const ProfileResult& result)
		{
			std::stringstream json;

			json << std::setprecision(3) << std::fixed;
			json << ",{";
			json << "\"cat\":\"function\",";
			json << "\"dur\":" << (result.ElapsedTime.count()) << ',';
			json << "\"name\":\"" << result.Name << "\",";
			json << "\"ph\":\"X\",";
			json << "\"pid\":0,";
			json << "\"tid\":" << result.ThreadID << ",";
			json << "\"ts\":" << result.Start.count();
			json << "}";

			std::lock_guard<std::mutex> lock(_mutex);
			if (_current_session != nullptr)
			{
				_output_stream << json.str();
				_output_stream.flush();
			}
		}

		static ProfileWriter& Get()
		{
			static ProfileWriter instance;
			return instance;
		}
	private:
		ProfileWriter()
			: _current_session(nullptr)
		{
		}

		~ProfileWriter()
		{
			EndSession();
		}		

		void WriteHeader()
		{
			_output_stream << "{\"otherData\": {},\"traceEvents\":[{}";
			_output_stream.flush();
		}

		void WriteFooter()
		{
			_output_stream << "]}";
			_output_stream.flush();
		}

		// Note: you must already own lock on m_Mutex before
		// calling InternalEndSession()
		void InternalEndSession()
		{
			if (_current_session != nullptr)
			{
				WriteFooter();
				_output_stream.close();
				delete _current_session;
				_current_session = nullptr;
			}
		}
	
    };

    class Profiler
    {
    public:
		Profiler(const char* name)
			: _name(name), _stopped(false)
		{
			_start_timepoint = std::chrono::steady_clock::now();
		}

		~Profiler()
		{
			if (!_stopped)
				Stop();
		}

		void Stop()
		{
			auto endTimepoint = std::chrono::steady_clock::now();
			auto highResStart = FloatingPointMicroseconds{ _start_timepoint.time_since_epoch() };
			auto elapsedTime = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch() - std::chrono::time_point_cast<std::chrono::microseconds>(_start_timepoint).time_since_epoch();

			ProfileWriter::Get().Write({ _name, highResStart, elapsedTime, std::this_thread::get_id() });

			_stopped = true;
		}
	private:
		const char* _name;
		std::chrono::time_point<std::chrono::steady_clock> _start_timepoint;
		bool _stopped;
    };

    namespace ProfilerUtils {

		template <size_t N>
		struct ChangeResult
		{
			char Data[N];
		};

		template <size_t N, size_t K>
		constexpr auto CleanupOutputString(const char(&expr)[N], const char(&remove)[K])
		{
			ChangeResult<N> result = {};

			size_t srcIndex = 0;
			size_t dstIndex = 0;
			while (srcIndex < N)
			{
				size_t matchIndex = 0;
				while (matchIndex < K - 1 && srcIndex + matchIndex < N - 1 && expr[srcIndex + matchIndex] == remove[matchIndex])
					matchIndex++;
				if (matchIndex == K - 1)
					srcIndex += matchIndex;
				result.Data[dstIndex++] = expr[srcIndex] == '"' ? '\'' : expr[srcIndex];
				srcIndex++;
			}
			return result;
		}
	}

}
#ifndef OPF_PROFILE_ACTIVATED
    #define OPF_PROFILE_ACTIVATED 0
#endif

#define OPF_PROFILE_ACTIVATED 0

#if OPF_PROFILE_ACTIVATED
	// Resolve which function signature macro will be used. Note that this only
	// is resolved when the (pre)compiler starts, so the syntax highlighting
	// could mark the wrong one in your editor!
	#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) || (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__)
		#define OPF_FUNC_SIG __PRETTY_FUNCTION__
	#elif defined(__DMC__) && (__DMC__ >= 0x810)
		#define OPF_FUNC_SIG __PRETTY_FUNCTION__
	#elif (defined(__FUNCSIG__) || (_MSC_VER))
		#define OPF_FUNC_SIG __FUNCSIG__
	#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) || (defined(__IBMCPP__) && (__IBMCPP__ >= 500))
		#define OPF_FUNC_SIG __FUNCTION__
	#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
		#define OPF_FUNC_SIG __FUNC__
	#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
		#define OPF_FUNC_SIG __func__
	#elif defined(__cplusplus) && (__cplusplus >= 201103)
		#define OPF_FUNC_SIG __func__
	#else
		#define OPF_FUNC_SIG "OPF_FUNC_SIG unknown!"
	#endif

	#define OPF_PROFILE_BEGIN_SESSION(name, filepath) ::debug_helper::ProfileWriter::Get().BeginSession(name, filepath)
	#define OPF_PROFILE_END_SESSION() ::debug_helper::ProfileWriter::Get().EndSession()
	#define OPF_PROFILE_SCOPE_LINE2(name, line) constexpr auto fixedName##line = ::debug_helper::ProfilerUtils::CleanupOutputString(name, "__cdecl ");\
											   ::debug_helper::Profiler timer##line(fixedName##line.Data)
	#define OPF_PROFILE_SCOPE_LINE(name, line) OPF_PROFILE_SCOPE_LINE2(name, line)
	#define OPF_PROFILE_SCOPE(name) OPF_PROFILE_SCOPE_LINE(name, __LINE__)
	#define OPF_PROFILE_FUNCTION() OPF_PROFILE_SCOPE(OPF_FUNC_SIG)
#else
	#define OPF_PROFILE_BEGIN_SESSION(name, filepath)
	#define OPF_PROFILE_END_SESSION()
	#define OPF_PROFILE_SCOPE(name)
	#define OPF_PROFILE_FUNCTION()
#endif