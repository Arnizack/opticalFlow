#pragma once
#include <memory>
#include "core/pyramid/IPyramid.h"

namespace cpu_backend
{
	template<class T>
	class Pyramid : public core::IPyramid<T>
	{
	public:
		Pyramid(const T next_level)
			: _current_level(next_level), _end_level(true), _next_Pyramid(nullptr)
		{}

		virtual T NextLevel() override
		{
			if (_end_level == true)
				return _current_level;

			//const T current = _current_level;
			
			_end_level = _next_Pyramid->IsEndLevel();
			_current_level = _next_Pyramid->GetCurrentLevel();

			if (_next_Pyramid->IsEndLevel() != true)
				_next_Pyramid = _next_Pyramid->GetNextPyramid();

			return _current_level;
		}

		virtual bool IsEndLevel() override
		{
			return _end_level;
		}

		void AddLevel(T next_level)
		{
			if (_end_level == true)
			{
				_end_level = false;

				_next_Pyramid = std::make_shared<Pyramid<T>>(Pyramid<T>(next_level));

			} else {
				_next_Pyramid->AddLevel(next_level);
			}
		}

		T GetCurrentLevel()
		{
			return _current_level;
		}

		std::shared_ptr<Pyramid<T>> GetNextPyramid()
		{
			return _next_Pyramid;
		}

	private:
		bool _end_level;
		T _current_level;
		std::shared_ptr<Pyramid<T>> _next_Pyramid;
	};

	/*
	* SHARED_PTR
	*/
	template<class T>
	class Pyramid<std::shared_ptr<T>> : public core::IPyramid<std::shared_ptr<T>>
	{
	public:
		Pyramid(const std::shared_ptr<T> next_level)
			: _current_level(std::make_shared<T>(*next_level)), _end_level(true), _next_Pyramid(nullptr)
		{}

		virtual std::shared_ptr<T> NextLevel() override
		{
			if (_end_level == true)
				return _current_level;

			const std::shared_ptr<T> current = std::make_shared<T>(*_current_level);

			_end_level = _next_Pyramid->IsEndLevel();
			_current_level = _next_Pyramid->GetCurrentLevel();

			if (_next_Pyramid->IsEndLevel() != true)
				_next_Pyramid = _next_Pyramid->GetNextPyramid();

			return current;
		}

		virtual bool IsEndLevel() override
		{
			return _end_level;
		}

		void AddLevel(std::shared_ptr<T> next_level)
		{
			if (_end_level == true)
			{
				_end_level = false;

				_next_Pyramid = std::make_shared<Pyramid<std::shared_ptr<T>>>(Pyramid<std::shared_ptr<T>>(next_level));

			}
			else {
				_next_Pyramid->AddLevel(next_level);
			}
		}

		std::shared_ptr<T> GetCurrentLevel()
		{
			return _current_level;
		}

		std::shared_ptr<Pyramid<std::shared_ptr<T>>> GetNextPyramid()
		{
			return _next_Pyramid;
		}

	private:
		bool _end_level;
		std::shared_ptr<T> _current_level;
		std::shared_ptr<Pyramid<std::shared_ptr<T>>> _next_Pyramid;
	};
}