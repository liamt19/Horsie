#pragma once

#include "../types.h"
#include <array>

namespace Horsie::Util 
{
	namespace ThingsIStoleFromStormphrax
	{
		template <typename T, nuint N, nuint... Ns>
		struct NDArrayImpl
		{
			using Type = std::array<typename NDArrayImpl<T, Ns...>::Type, N>;
		};

		template <typename T, nuint N>
		struct NDArrayImpl<T, N>
		{
			using Type = std::array<T, N>;
		};
	}

	template <typename T, nuint... Ns>
	using NDArray = typename ThingsIStoleFromStormphrax::NDArrayImpl<T, Ns...>::Type;
}