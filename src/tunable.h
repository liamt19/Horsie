#pragma once

#ifndef TUNABLE_H
#define TUNABLE_H

#include "defs.h"

#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

struct TunableOption {
    std::string Name;
    i32 DefaultValue;
    i32 CurrentValue;
    i32 MinValue;
    i32 MaxValue;
    double Step;

    TunableOption(const std::string& name, i32 v, i32 min, i32 max, double step) :
        Name(name),
        DefaultValue(v),
        CurrentValue(v),
        MinValue(min),
        MaxValue(max),
        Step(step) {
    }

    operator i32() const { return CurrentValue; }

    TunableOption& operator=(i32 newV) {
        if (newV < MinValue || newV > MaxValue) {
            throw std::out_of_range("Tunable assignment out of range");
        }

        CurrentValue = newV;

        return *this;
    }
};


inline std::ostream& operator<<(std::ostream& os, const TunableOption& opt) {
    os << "option name " << opt.Name << " type ";

    if (opt.Name.rfind("UCI_", 0) == 0) {
        os << "check default " << (opt.DefaultValue ? "true" : "false");
    }
    else {
        os << "spin default " << opt.DefaultValue << " min " << opt.MinValue << " max " << opt.MaxValue;
    }

    return os;
}


inline std::vector<TunableOption>& GetUCIOptions() {
    static auto opts = []
    {
        std::vector<TunableOption> opts{};
        opts.reserve(128);
        return opts;
    }();

    return opts;
}

inline TunableOption* FindUCIOption(const std::string& name) {
    auto& opts = GetUCIOptions(); 
    for (auto& opt : opts) {
        auto lowerName = opt.Name;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), [](auto c) { return std::tolower(c); });
        if (lowerName == name)
            return &opt;
    }

    return nullptr;
}

inline TunableOption& AddUCIOption(const std::string& name, i32 v, i32 min, i32 max, double step) {
    auto& opts = GetUCIOptions();
    auto lowerName = name;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), [](auto c) { return std::tolower(c); });
    return opts.emplace_back(TunableOption{ name, v, min, max, step });
}

inline TunableOption& AddUCIOption(const std::string& name, i32 v, i32 min, i32 max) {
    return AddUCIOption(name, v, min, max, std::max(0.5, (max - min) / 20.0));
}

inline TunableOption& AddUCIOption(const std::string& name, i32 v) {
    auto min = static_cast<i32>(std::round(v * (1 - 0.45)));
    auto max = static_cast<i32>(std::round(v * (1 + 0.45)));
    return AddUCIOption(name, v, min, max);
}


#define UCI_OPTION_SPIN(Name, Default) \
    inline TunableOption& Name = AddUCIOption(#Name, Default, false, true, 1);

#define UCI_OPTION(Name, Default) \
    inline TunableOption& Name = AddUCIOption(#Name, Default);

#define UCI_OPTION_SPECIAL(Name, Default, Min, Max) \
    inline TunableOption& Name = AddUCIOption(#Name, Default, Min, Max);

#define UCI_OPTION_CUSTOM(Name, Default, Min, Max) \
    inline TunableOption& Name = AddUCIOption(#Name, Default, Min, Max);


#endif // !TUNABLE_H
