#pragma once

#ifndef TUNABLE_H
#define TUNABLE_H

#include "defs.h"

#include <string>
#include <cmath>
#include <vector>
#include <algorithm>


struct TunableParam {
    std::string Name;
    i32 DefaultValue;
    i32 CurrentValue;
    i32 MinValue;
    i32 MaxValue;
    double Step;

    TunableParam(const std::string& name, i32 v, i32 min, i32 max, double step) :
        Name(name),
        DefaultValue(v),
        CurrentValue(v),
        MinValue(min),
        MaxValue(max),
        Step(step) {
    }

    operator i32() const { return CurrentValue; }

    TunableParam& operator=(i32 newV) {
        if (newV < MinValue || newV > MaxValue) {
            throw std::out_of_range("Tunable assignment out of range");
        }

        CurrentValue = newV;

        return *this;
    }
};


inline std::vector<TunableParam>& tunableParams() {
    static auto params = []
    {
        std::vector<TunableParam> params{};
        params.reserve(128);
        return params;
    }();

    return params;
}

inline TunableParam* lookupTunableParam(const std::string& name) {
    for (auto& param : tunableParams()) {
        auto lowerName = param.Name;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), [](auto c) { return std::tolower(c); });
        if (lowerName == name)
            return &param;
    }

    return nullptr;
}

inline TunableParam& addTunableParam(const std::string& name, i32 v, i32 min, i32 max, double step) {
    auto& params = tunableParams();
    auto lowerName = name;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), [](auto c) { return std::tolower(c); });
    return params.emplace_back(TunableParam{ name, v, min, max, step });
}

inline TunableParam& addTunableParam(const std::string& name, i32 v, i32 min, i32 max) {
    return addTunableParam(name, v, min, max, std::max(0.5, (max - min) / 20.0));
}

inline TunableParam& addTunableParam(const std::string& name, i32 v) {
    auto min = static_cast<i32>(std::round(v * (1 - 0.45)));
    auto max = static_cast<i32>(std::round(v * (1 + 0.45)));
    return addTunableParam(name, v, min, max);
}


#define TUNABLE_PARAM(Name, Default) \
        inline TunableParam& Name = addTunableParam(#Name, Default);

#define TUNABLE_PARAM_SPECIAL(Name, Default, Min, Max) \
		inline TunableParam& Name = addTunableParam(#Name, Default, Min, Max);

#define TUNABLE_PARAM_CUSTOM(Name, Default, Min, Max) \
		inline TunableParam& Name = addTunableParam(#Name, Default, Min, Max);


#endif // !TUNABLE_H
