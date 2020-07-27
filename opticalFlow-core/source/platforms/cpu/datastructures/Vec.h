#pragma once

#include <array>
#include <algorithm>
#include <initializer_list>
#include <iterator>


namespace cpu
{
    template<class T, size_t dim>
    struct Vec
    {
        std::array<T, dim> data;

        //initializer list
        template<typename ... Value>
        Vec(Value ... values)
            :data{ {values ... } }
        {}

        Vec() = default;

        inline T& operator[](const int& i)
        {
            return this->data[i];
        }

        inline const T& operator[] (const int& i) const
        {
            return this->data[i];
        }

        inline Vec& operator=(const Vec& other)
        {
            std::copy(other.cbegin(), other.cend(), begin());
            return *this;
        }

        inline Vec operator+(const Vec& a) const
        {
            Vec result = *this;
            result += a;
            return result;
        }

        inline void operator+=(const Vec& a)
        {
            for (size_t i = 0; i < dim; i++)
            {
                data[i] += a[i];
            }
        }

        inline Vec operator-(const Vec& a) const
        {
            Vec result = *this;
            result -= a;
            return result;
        }

        inline void operator-=(const Vec& a)
        {
            for (size_t i = 0; i < dim; i++)
            {
                data[i] -= a[i];
            }
        }

        inline Vec operator*(const T& s)
        {
            Vec result = *this;
            result *= s;
            return s;
        }

        inline void operator*=(const T& s)
        {
            for (size_t i = 0; i < dim; i++)
            {
                data[i] *= s;
            }
        }

        inline Vec operator/(const T& s)
        {
            Vec result = *this;
            result /= s;
            return s;
        }

        inline void operator/=(const T& s)
        {
            for (size_t i = 0; i < dim; i++)
            {
                data[i] /= s;
            }
        }

        inline auto begin() noexcept
        {
            return data.begin();
        }

        inline auto end() noexcept
        {
            return data.end();
        }

        inline auto cbegin() const noexcept
        {
            return data.cbegin();
        }

        inline auto cend() const noexcept
        {
            return data.cend();
        }
    };
}