#pragma once

#include "../types.h"
#include <array>

namespace Horsie::Util 
{
	namespace ThingsIStoleFromStormphrax
	{
		template <typename T, usize N, usize... Ns>
		struct NDArrayImpl
		{
			using Type = std::array<typename NDArrayImpl<T, Ns...>::Type, N>;
		};

		template <typename T, usize N>
		struct NDArrayImpl<T, N>
		{
			using Type = std::array<T, N>;
		};
	}

	template <typename T, usize... Ns>
	using NDArray = typename ThingsIStoleFromStormphrax::NDArrayImpl<T, Ns...>::Type;
}