#pragma once
#include "core\linalg\IArithmeticBasic.h"

//#include "../Array.h"
#include "../ArrayFactory.h"

#include <math.h>
#include <omp.h>
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
		using PtrArrayFactory = std::shared_ptr<core::IArrayFactory<InnerTyp, DimCount>>;

	public:
		ArithmeticBasic(const PtrArrayFactory factory) 
			: _factory(std::dynamic_pointer_cast<ArrayFactory<InnerTyp, DimCount>>(factory))
		{}


		//a+b
		virtual PtrVector Add(const PtrVector a, const PtrVector b) override
		{
			//a+b
			const size_t size = a->Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			auto out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(_factory->Zeros(a_obj->Shape));

			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] + (*b_obj)[i];
			}

			return out;
		}

		//x = a+b
		virtual void AddTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a+b
			const size_t size = a->Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			std::shared_ptr<Array<InnerTyp, DimCount>> out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(x); 
			
			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] + (*b_obj)[i];
			}

			return;
		}

		//a-b
		virtual PtrVector Sub(const PtrVector a, const PtrVector b) override
		{
			//a-b
			const size_t size = a->Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			auto out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(_factory->Zeros(a_obj->Shape));

			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] - (*b_obj)[i];
			}

			return out;
		}

		//x = a-b
		virtual void SubTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a+b
			const size_t size = a->Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			std::shared_ptr<Array<InnerTyp, DimCount>> out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(x);

			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] - (*b_obj)[i];
			}

			return;
		}

		//a*b
		virtual PtrVector Mul(const PtrVector a, const PtrVector b) override
		{
			//a*b
			const size_t size = a->Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			auto out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(_factory->Zeros(a_obj->Shape));

			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] * (*b_obj)[i];
			}

			return out;
		}

		//x = a*b
		virtual void MulTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a*b
			const size_t size = a->Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			std::shared_ptr<Array<InnerTyp, DimCount>> out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(x);

			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (*a_obj)[i] * (*b_obj)[i];
			}

			return;
		}

		//a/b
		virtual PtrVector Div(const PtrVector a, const PtrVector b) override
		{
			//a/b
			const size_t size = a->Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			auto out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(_factory->Zeros(a_obj->Shape));

			#pragma omp parallel for
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

			return out;
		}

		//x = a/b
		virtual void DivTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//x = a/b
			const size_t size = a->Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			std::shared_ptr<Array<InnerTyp, DimCount>> out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(x);

			#pragma omp parallel for
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

			return;
		}
		
		//a**b
		virtual PtrVector Pow(const PtrVector a, const PtrVector b) override
		{
			//a^b
			const size_t size = a->Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			auto out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(_factory->Zeros(a_obj->Shape));

			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (InnerTyp) pow((double)(*a_obj)[i], (double)(*b_obj)[i]);
			}

			return out;
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const PtrVector b) override
		{
			//a^b
			const size_t size = a->Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);
			std::shared_ptr<Array<InnerTyp, DimCount>> b_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(b);

			std::shared_ptr<Array<InnerTyp, DimCount>> out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(x);

			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (InnerTyp)pow((double)(*a_obj)[i], (double)(*b_obj)[i]);
			}

			return;
		}

		//a**b
		virtual PtrVector Pow(const PtrVector a, const double& b) override
		{
			//a^b
			const size_t size = a->Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);

			auto out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(_factory->Zeros(a_obj->Shape));

			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (InnerTyp)pow((double)(*a_obj)[i], b);
			}

			return out;
		}

		//x = a**b
		virtual void PowTo(PtrVector x, const PtrVector a, const double& b) override
		{
			//a^b
			const size_t size = a->Size();

			std::shared_ptr<Array<InnerTyp, DimCount>> a_obj = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(a);

			std::shared_ptr<Array<InnerTyp, DimCount>> out = std::dynamic_pointer_cast<Array<InnerTyp, DimCount>>(x);

			#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				(*out)[i] = (InnerTyp)pow((double)(*a_obj)[i], b);
			}

			return;
		}

		private:
			std::shared_ptr<ArrayFactory<InnerTyp, DimCount>> _factory;
	};
	
	/*
	* DOUBLE
	*/
	
	//template<size_t DimCount>
	//class ArithmeticBasic<double, DimCount> : public core::IArithmeticBasic<double, DimCount>
	//{
	//	using PtrVector = std::shared_ptr<core::IArray<double, DimCount>>;

	//public:

	//	//a+b
	//	virtual PtrVector Add(const PtrVector a, const PtrVector b) override
	//	{
	//		//a+b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);

	//		std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, (*b)));

	//		cblas_daxpby(size, 1, &(*a_obj)[0], 1, 1, &(*out)[0], 1);

	//		return out;
	//	}

	//	//x = a+b
	//	virtual void AddTo(PtrVector x, const PtrVector a, const PtrVector b) override
	//	{
	//		//x = a+b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);
	//		std::shared_ptr<Array<double, DimCount>> b_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(b);

	//		std::shared_ptr<Array<double, DimCount>> out = std::dynamic_pointer_cast<Array<double, DimCount>>(x);

	//		for (size_t i = 0; i < b_obj->Size(); i++)
	//		{
	//			(*out)[i] = (*b_obj)[i]; //(*out)[i] = (*a_obj)[i] + (*b_obj)[i];
	//		}

	//		cblas_daxpby(size, 1, &(*a_obj)[0], 1, 1, &(*out)[0], 1);

	//		return;
	//	}

	//	//a-b
	//	virtual PtrVector Sub(const PtrVector a, const PtrVector b) override
	//	{
	//		//a-b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);

	//		std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, (*b)));

	//		cblas_daxpby(size, 1, &(*a_obj)[0], 1, -1, &(*out)[0], 1);

	//		return out;
	//	}

	//	//x = a-b
	//	virtual void SubTo(PtrVector x, const PtrVector a, const PtrVector b) override
	//	{
	//		//x = a-b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);
	//		std::shared_ptr<Array<double, DimCount>> b_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(b);

	//		std::shared_ptr<Array<double, DimCount>> out = std::dynamic_pointer_cast<Array<double, DimCount>>(x);

	//		for (size_t i = 0; i < b_obj->Size(); i++)
	//		{
	//			(*out)[i] = (*b_obj)[i]; // (*out)[i] = (*a_obj)[i] - (*b_obj)[i];
	//		}

	//		cblas_daxpby(size, 1, &(*a_obj)[0], 1, -1, &(*out)[0], 1);

	//		return;
	//	}

	//	//a*b
	//	virtual PtrVector Mul(const PtrVector a, const PtrVector b) override
	//	{
	//		//a*b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);
	//		std::shared_ptr<Array<double, DimCount>> b_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(b);

	//		std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, size));

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = (*a_obj)[i] * (*b_obj)[i];
	//		}

	//		return out;
	//	}

	//	//x = a*b
	//	virtual void MulTo(PtrVector x, const PtrVector a, const PtrVector b) override
	//	{
	//		//x = a*b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);
	//		std::shared_ptr<Array<double, DimCount>> b_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(b);

	//		std::shared_ptr<Array<double, DimCount>> out = std::dynamic_pointer_cast<Array<double, DimCount>>(x);

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = (*a_obj)[i] * (*b_obj)[i];
	//		}

	//		return;
	//	}

	//	//a/b
	//	virtual PtrVector Div(const PtrVector a, const PtrVector b) override
	//	{
	//		//a/b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);
	//		std::shared_ptr<Array<double, DimCount>> b_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(b);

	//		std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, size));

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = (*a_obj)[i] / (*b_obj)[i];
	//		}

	//		return out;
	//	}

	//	//x = a/b
	//	virtual void DivTo(PtrVector x, const PtrVector a, const PtrVector b) override
	//	{
	//		//x = a/b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);
	//		std::shared_ptr<Array<double, DimCount>> b_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(b);

	//		std::shared_ptr<Array<double, DimCount>> out = std::dynamic_pointer_cast<Array<double, DimCount>>(x);

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = (*a_obj)[i] / (*b_obj)[i];
	//		}

	//		return;
	//	}

	//	//a**b
	//	virtual PtrVector Pow(const PtrVector a, const PtrVector b) override
	//	{
	//		//a^b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);
	//		std::shared_ptr<Array<double, DimCount>> b_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(b);

	//		std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, size));

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = pow((*a_obj)[i], (*b_obj)[i]);
	//		}

	//		return out;
	//	}

	//	//x = a**b
	//	virtual void PowTo(PtrVector x, const PtrVector a, const PtrVector b) override
	//	{
	//		//x = a^b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);
	//		std::shared_ptr<Array<double, DimCount>> b_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(b);

	//		std::shared_ptr<Array<double, DimCount>> out = std::dynamic_pointer_cast<Array<double, DimCount>>(x);

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = pow((*a_obj)[i], (*b_obj)[i]);
	//		}

	//		return;
	//	}

	//	//a**b
	//	virtual PtrVector Pow(const PtrVector a, const double& b) override
	//	{
	//		//a^b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);

	//		std::shared_ptr<Array<double, DimCount>> out = std::make_shared<Array<double, DimCount>>(Array<double, DimCount>((*a).Shape, size));

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = pow((*a_obj)[i], b);
	//		}

	//		return out;
	//	}

	//	//x = a**b
	//	virtual void PowTo(PtrVector x, const PtrVector a, const double& b) override
	//	{
	//		//a^b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<double, DimCount>> a_obj = std::dynamic_pointer_cast<Array<double, DimCount>>(a);

	//		std::shared_ptr<Array<double, DimCount>> out = std::dynamic_pointer_cast<Array<double, DimCount>>(x);

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = pow((*a_obj)[i], b);
	//		}

	//		return;
	//	}
	//};

	/*
	* FLOAT
	*/
	//template<size_t DimCount>
	//class ArithmeticBasic<float, DimCount> : public core::IArithmeticBasic<float, DimCount>
	//{
	//	using PtrVector = std::shared_ptr<core::IArray<float, DimCount>>;

	//public:

	//	//a+b
	//	virtual PtrVector Add(const PtrVector a, const PtrVector b) override
	//	{
	//		//a+b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);

	//		std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, (*b)));

	//		cblas_saxpby(size, 1, &(*a_obj)[0], 1, 1, &(*out)[0], 1);

	//		return std::static_pointer_cast<core::IArray<float, DimCount>>(out);
	//	}

	//	//x = a+b
	//	virtual void AddTo(PtrVector x, const PtrVector a, const PtrVector b) override
	//	{
	//		//x = a+b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);
	//		std::shared_ptr<Array<float, DimCount>> b_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(b);

	//		std::shared_ptr<Array<float, DimCount>> out = std::dynamic_pointer_cast<Array<float, DimCount>>(x);

	//		for (size_t i = 0; i < b_obj->Size(); i++)
	//		{
	//			(*out)[i] = (*b_obj)[i]; // (*out)[i] = (*a_obj)[i] + (*b_obj)[i];
	//		}

	//		cblas_saxpby(size, 1, &(*a_obj)[0], 1, 1, &(*out)[0], 1);

	//		return;
	//	}

	//	//a-b
	//	virtual PtrVector Sub(const PtrVector a, const PtrVector b) override
	//	{
	//		//a-b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);

	//		std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, (*b)));

	//		cblas_saxpby(size, 1, &(*a_obj)[0], 1, -1, &(*out)[0], 1);

	//		return out;
	//	}

	//	//x = a-b
	//	virtual void SubTo(PtrVector x, const PtrVector a, const PtrVector b) override
	//	{
	//		//x = a-b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);
	//		std::shared_ptr<Array<float, DimCount>> b_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(b);

	//		std::shared_ptr<Array<float, DimCount>> out = std::dynamic_pointer_cast<Array<float, DimCount>>(x);

	//		for (size_t i = 0; i < b_obj->Size(); i++)
	//		{
	//			(*out)[i] = (*b_obj)[i]; // (*out)[i] = (*a_obj)[i] - (*b_obj)[i];
	//		}

	//		cblas_saxpby(size, 1, &(*a_obj)[0], 1, -1, &(*out)[0], 1);

	//		return;
	//	}

	//	//a*b
	//	virtual PtrVector Mul(const PtrVector a, const PtrVector b) override
	//	{
	//		//a*b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);
	//		std::shared_ptr<Array<float, DimCount>> b_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(b);

	//		std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, size));

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = (*a_obj)[i] * (*b_obj)[i];
	//		}

	//		return out;
	//	}

	//	//x = a*b
	//	virtual void MulTo(PtrVector x, const PtrVector a, const PtrVector b) override
	//	{
	//		//x = a*b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);
	//		std::shared_ptr<Array<float, DimCount>> b_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(b);

	//		std::shared_ptr<Array<float, DimCount>> out = std::dynamic_pointer_cast<Array<float, DimCount>>(x);

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = (*a_obj)[i] * (*b_obj)[i];
	//		}

	//		return;
	//	}

	//	//a/b
	//	virtual PtrVector Div(const PtrVector a, const PtrVector b) override
	//	{
	//		//a/b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);
	//		std::shared_ptr<Array<float, DimCount>> b_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(b);

	//		std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, size));

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = (*a_obj)[i] / (*b_obj)[i];
	//		}

	//		return out;
	//	}

	//	//x = a/b
	//	virtual void DivTo(PtrVector x, const PtrVector a, const PtrVector b) override
	//	{
	//		//x = a/b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);
	//		std::shared_ptr<Array<float, DimCount>> b_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(b);

	//		std::shared_ptr<Array<float, DimCount>> out = std::dynamic_pointer_cast<Array<float, DimCount>>(x);

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = (*a_obj)[i] / (*b_obj)[i];
	//		}

	//		return;
	//	}

	//	//a**b
	//	virtual PtrVector Pow(const PtrVector a, const PtrVector b) override
	//	{
	//		//a^b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);
	//		std::shared_ptr<Array<float, DimCount>> b_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(b);

	//		std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, size));

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = pow((*a_obj)[i], (*b_obj)[i]);
	//		}

	//		return out;
	//	}

	//	//x = a**b
	//	virtual void PowTo(PtrVector x, const PtrVector a, const PtrVector b) override
	//	{
	//		//x = a^b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);
	//		std::shared_ptr<Array<float, DimCount>> b_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(b);

	//		std::shared_ptr<Array<float, DimCount>> out = std::dynamic_pointer_cast<Array<float, DimCount>>(x);

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = pow((*a_obj)[i], (*b_obj)[i]);
	//		}

	//		return;
	//	}

	//	//a**b
	//	virtual PtrVector Pow(const PtrVector a, const double& b) override
	//	{
	//		//a^b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);

	//		std::shared_ptr<Array<float, DimCount>> out = std::make_shared<Array<float, DimCount>>(Array<float, DimCount>((*a).Shape, size));

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = pow((*a_obj)[i], b);
	//		}

	//		return out;
	//	}

	//	//x = a**b
	//	virtual void PowTo(PtrVector x, const PtrVector a, const double& b) override
	//	{
	//		//a^b
	//		const size_t size = a->Size();

	//		std::shared_ptr<Array<float, DimCount>> a_obj = std::dynamic_pointer_cast<Array<float, DimCount>>(a);

	//		std::shared_ptr<Array<float, DimCount>> out = std::dynamic_pointer_cast<Array<float, DimCount>>(x);

	//		for (int i = 0; i < size; i++)
	//		{
	//			(*out)[i] = pow((*a_obj)[i], b);
	//		}

	//		return;
	//	}
	//};

}