#pragma once
#include "core\linalg\IArithmeticBasic.h"

#include "../Array.h"

#include <math.h>
#include <cblas.h>

namespace cpu_backend
{

	/*
	* All Types
	*/
	template<class InnerTyp, size_t DimCount>
	class ArithmeticBasic : public core::IArithmeticBasic<InnerTyp, DimCount>
	{
		using PtrVector = std::shared_ptr<core::IArray<InnerTyp, DimCount>>;

	public:

		//a+b
		virtual PtrVector Add(const PtrVector a, const PtrVector b) override
		{
			//a+b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			Array<InnerTyp, DimCount> out(a.get()->Shape, size);

			for (int i = 0; i < size; i++)
			{
				out[i] = (*a_obj)[i] + (*b_obj)[i];
			}

			return std::make_shared<Array<InnerTyp, DimCount>>(out);
		}

		//x = a+b
		virtual void AddTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a+b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			std::shared_ptr<Array<InnerTyp, DimCount>> out = std::make_shared<Array<InnerTyp, DimCount>>(Array<InnerTyp, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] + (*b_obj)[i];
			}

			x = std::static_pointer_cast<core::IArray<InnerTyp, DimCount>>(out);

			return;
		}

		//a-b
		virtual PtrVector Sub(const PtrVector a, const PtrVector b) override
		{
			//a-b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			Array<InnerTyp, DimCount> out(a.get()->Shape, size);

			for (int i = 0; i < size; i++)
			{
				out[i] = (*a_obj)[i] - (*b_obj)[i];
			}

			return std::make_shared<Array<InnerTyp, DimCount>>(out);
		}

		//x = a-b
		virtual void SubTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a+b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			std::shared_ptr<Array<InnerTyp, DimCount>> out = std::make_shared<Array<InnerTyp, DimCount>>(Array<InnerTyp, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] - (*b_obj)[i];
			}

			x = std::static_pointer_cast<core::IArray<InnerTyp, DimCount>>(out);

			return;
		}

		//a*b
		virtual PtrVector Mul(const PtrVector a, const PtrVector b) override
		{
			//a*b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			Array<InnerTyp, DimCount> out(a.get()->Shape, size);

			for (int i = 0; i < size; i++)
			{
				out[i] = (*a_obj)[i] * (*b_obj)[i];
			}

			return std::make_shared<Array<InnerTyp, DimCount>>(out);
		}

		//x = a*b
		virtual void MulTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a*b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			std::shared_ptr<Array<InnerTyp, DimCount>> out = std::make_shared<Array<InnerTyp, DimCount>>(Array<InnerTyp, DimCount>((*a).Shape));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] * (*b_obj)[i];
			}

			x = std::static_pointer_cast<core::IArray<InnerTyp, DimCount>>(out);

			return;
		}

		//a/b
		virtual PtrVector Div(const PtrVector a, const PtrVector b) override
		{
			//a/b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			Array<InnerTyp, DimCount> out(a.get()->Shape, size);

			for (int i = 0; i < size; i++)
			{
				if ((*b_obj)[i] == 0)
				{
					out[i] = (InnerTyp)INFINITY;
				}
				else
				{
					out[i] = (InnerTyp)((*a_obj)[i] / (*b_obj)[i]);
				}
			}

			return std::make_shared<Array<InnerTyp, DimCount>>(out);
		}

		//x = a/b
		virtual void DivTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a/b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			std::shared_ptr<Array<InnerTyp, DimCount>> out = std::make_shared<Array<InnerTyp, DimCount>>(Array<InnerTyp, DimCount>((*a).Shape));

			for (int i = 0; i < size; i++)
			{
				if ((*b_obj)[i] == 0)
				{
					(*out)[i] = (InnerTyp)INFINITY;
				}
				else
				{
					(*out)[i] = (InnerTyp)((*a_obj)[i] / (*b_obj)[i]);
				}
			}

			x = std::static_pointer_cast<core::IArray<InnerTyp, DimCount>>(out);

			return;
		}
		
		//a**b
		virtual PtrVector Pow(const PtrVector a, const PtrVector b) override
		{
			//a^b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			Array<InnerTyp, DimCount> out(a.get()->Shape, size);

			for (int i = 0; i < size; i++)
			{
				out[i] = (InnerTyp) pow((double)(*a_obj)[i], (double)(*b_obj)[i]);
			}

			return std::make_shared<Array<InnerTyp, DimCount>>(out);
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//a^b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			std::shared_ptr<Array<InnerTyp, DimCount>> out = std::make_shared<Array<InnerTyp, DimCount>>(Array<InnerTyp, DimCount>((*a).Shape) );

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (InnerTyp)pow((double)(*a_obj)[i], (double)(*b_obj)[i]);
			}

			x = std::static_pointer_cast<core::IArray<InnerTyp, DimCount>>(out);

			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const double& b) override
		{
			//a^b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);

			double b_conv = (double)b;

			Array<InnerTyp, DimCount> out(a.get()->Shape, size);

			for (int i = 0; i < size; i++)
			{
				out[i] = (InnerTyp)pow((double)(*a_obj)[i], b_conv);
			}

			return std::make_shared<Array<InnerTyp, DimCount>>(out);
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const double& b) override
		{
			//a^b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);

			double b_conv = (double)b;

			std::shared_ptr<Array<InnerTyp, DimCount>> out = std::make_shared<Array<InnerTyp, DimCount>>(Array<InnerTyp, DimCount>((*a).Shape) );

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (InnerTyp)pow((double)(*a_obj)[i], b_conv);
			}

			x = std::static_pointer_cast<core::IArray<InnerTyp, DimCount>>(out);

			return;
		}
	};
	
	/*
	* DOUBLE
	*/
	
	template<size_t DimCount>
	class ArithmeticBasic<double, DimCount> : public core::IArithmeticBasic<double, DimCount>
	{
		using PtrVector = std::shared_ptr<core::IArray<double, DimCount>>;

	public:

		//a+b
		virtual PtrVector Add(const PtrVector a, const PtrVector b) override
		{
			//a+b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);

			std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, (*b)));

			cblas_daxpby(size, 1, &(*a_obj)[0], 1, 1, &(*out)[0], 1);

			return std::static_pointer_cast<core::IArray<double, DimCount>>(out);
		}

		//x = a+b
		virtual void AddTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a+b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);

			std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, (*b)));

			cblas_daxpby(size, 1, &(*a_obj)[0], 1, 1, &(*out)[0], 1);

			x = std::static_pointer_cast<core::IArray<double, DimCount>>(out);

			return;
		}

		//a-b
		virtual PtrVector Sub(const PtrVector a, const PtrVector b) override
		{
			//a-b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);

			std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, (*b)));

			cblas_daxpby(size, 1, &(*a_obj)[0], 1, -1, &(*out)[0], 1);

			return std::static_pointer_cast<core::IArray<double, DimCount>>(out);
		}

		//x = a-b
		virtual void SubTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a-b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);

			std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, (*b)));

			cblas_daxpby(size, 1, &(*a_obj)[0], 1, -1, &(*out)[0], 1);

			x = std::static_pointer_cast<core::IArray<double, DimCount>>(out);

			return;
		}

		//a*b
		virtual PtrVector Mul(const PtrVector a, const PtrVector b) override
		{
			//a*b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);
			std::shared_ptr<Array<double, DimCount>> b_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(b);

			std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] * (*b_obj)[i];
			}

			return std::static_pointer_cast<core::IArray<double, DimCount>>(out);
		}

		//x = a*b
		virtual void MulTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a*b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);
			std::shared_ptr<Array<double, DimCount>> b_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(b);

			std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] * (*b_obj)[i];
			}

			x = std::static_pointer_cast<core::IArray<double, DimCount>>(out);
			return;
		}

		//a/b
		virtual PtrVector Div(const PtrVector a, const PtrVector b) override
		{
			//a/b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);
			std::shared_ptr<Array<double, DimCount>> b_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(b);

			std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] / (*b_obj)[i];
			}

			return std::static_pointer_cast<core::IArray<double, DimCount>>(out);
		}

		//x = a/b
		virtual void DivTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a/b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);
			std::shared_ptr<Array<double, DimCount>> b_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(b);

			std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] / (*b_obj)[i];
			}

			x = std::static_pointer_cast<core::IArray<double, DimCount>>(out);
			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const PtrVector b) override
		{
			//a^b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);
			std::shared_ptr<Array<double, DimCount>> b_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(b);

			std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = pow((*a_obj)[i], (*b_obj)[i]);
			}

			return std::static_pointer_cast<core::IArray<double, DimCount>>(out);
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a^b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);
			std::shared_ptr<Array<double, DimCount>> b_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(b);

			std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = pow((*a_obj)[i], (*b_obj)[i]);
			}

			x = std::static_pointer_cast<core::IArray<double, DimCount>>(out);
			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const double& b) override
		{
			//a^b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);

			std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = pow((*a_obj)[i], b);
			}

			return std::static_pointer_cast<core::IArray<double, DimCount>>(out);
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const double& b) override
		{
			//a^b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);

			std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = pow((*a_obj)[i], b);
			}

			x = std::static_pointer_cast<core::IArray<double, DimCount>>(out);
			return;
		}
	};

	/*
	* FLOAT
	*/
	template<size_t DimCount>
	class ArithmeticBasic<float, DimCount> : public core::IArithmeticBasic<float, DimCount>
	{
		using PtrVector = std::shared_ptr<core::IArray<float, DimCount>>;

	public:

		//a+b
		virtual PtrVector Add(const PtrVector a, const PtrVector b) override
		{
			//a+b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);

			std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, (*b)));

			cblas_saxpby(size, 1, &(*a_obj)[0], 1, 1, &(*out)[0], 1);

			return std::static_pointer_cast<core::IArray<float, DimCount>>(out);
		}

		//x = a+b
		virtual void AddTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a+b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);

			std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, (*b)));

			cblas_saxpby(size, 1, &(*a_obj)[0], 1, 1, &(*out)[0], 1);

			x = std::static_pointer_cast<core::IArray<float, DimCount>>(out);

			return;
		}

		//a-b
		virtual PtrVector Sub(const PtrVector a, const PtrVector b) override
		{
			//a-b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);

			std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, (*b)));

			cblas_saxpby(size, 1, &(*a_obj)[0], 1, -1, &(*out)[0], 1);

			return std::static_pointer_cast<core::IArray<float, DimCount>>(out);
		}

		//x = a-b
		virtual void SubTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a-b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);

			std::shared_ptr<Array<float, DimCount>> out 
				= std::make_shared<Array<float, DimCount>>
				(Array<float, DimCount> ((*a).Shape, *b) );

			cblas_saxpby(size, 1, &(*a_obj)[0], 1, -1, &(*out)[0], 1);

			x = std::static_pointer_cast<core::IArray<float, DimCount>>(out);

			return;
		}

		//a*b
		virtual PtrVector Mul(const PtrVector a, const PtrVector b) override
		{
			//a*b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);
			std::shared_ptr<Array<float, DimCount>> b_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(b);

			std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] * (*b_obj)[i];
			}

			return std::static_pointer_cast<core::IArray<float, DimCount>>(out);
		}

		//x = a*b
		virtual void MulTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a*b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);
			std::shared_ptr<Array<float, DimCount>> b_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(b);

			std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] * (*b_obj)[i];
			}

			x = std::static_pointer_cast<core::IArray<float, DimCount>>(out);
			return;
		}

		//a/b
		virtual PtrVector Div(const PtrVector a, const PtrVector b) override
		{
			//a/b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);
			std::shared_ptr<Array<float, DimCount>> b_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(b);

			std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] / (*b_obj)[i];
			}

			return std::static_pointer_cast<core::IArray<float, DimCount>>(out);
		}

		//x = a/b
		virtual void DivTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a/b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);
			std::shared_ptr<Array<float, DimCount>> b_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(b);

			std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] / (*b_obj)[i];
			}

			x = std::static_pointer_cast<core::IArray<float, DimCount>>(out);
			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const PtrVector b) override
		{
			//a^b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);
			std::shared_ptr<Array<float, DimCount>> b_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(b);

			std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = pow((*a_obj)[i], (*b_obj)[i]);
			}

			return std::static_pointer_cast<core::IArray<float, DimCount>>(out);
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a^b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);
			std::shared_ptr<Array<float, DimCount>> b_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(b);

			std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = pow((*a_obj)[i], (*b_obj)[i]);
			}

			x = std::static_pointer_cast<core::IArray<float, DimCount>>(out);
			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const double& b) override
		{
			//a^b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);

			std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = pow((*a_obj)[i], b);
			}

			return std::static_pointer_cast<core::IArray<float, DimCount>>(out);
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const double& b) override
		{
			//a^b
			const size_t size = (*a).Size();

			std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);

			std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, size));

			for (int i = 0; i < size; i++)
			{
				(*out)[i] = pow((*a_obj)[i], b);
			}

			x = std::static_pointer_cast<core::IArray<float, DimCount>>(out);
			return;
		}
	};

}