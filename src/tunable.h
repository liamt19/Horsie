#pragma once

#ifndef TUNABLE_H
#define TUNABLE_H

#include "defs.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

struct TunableOption {
    std::string Name;
    i32 DefaultValue;
    i32 CurrentValue;
    i32 MinValue;
    i32 MaxValue;
    double Step;
    bool HideTune;

    TunableOption(const std::string& name, i32 v, i32 min, i32 max, double step, bool hideTune = false) :
        Name(name),
        DefaultValue(v),
        CurrentValue(v),
        MinValue(min),
        MaxValue(max),
        Step(step),
        HideTune(hideTune) {
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

inline TunableOption& AddUCIOption(const std::string& name, i32 v, i32 min, i32 max, double step, bool hideTune = false) {
    auto& opts = GetUCIOptions();
    auto lowerName = name;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), [](auto c) { return std::tolower(c); });
    return opts.emplace_back(TunableOption{ name, v, min, max, step, hideTune });
}

inline TunableOption& AddUCIOption(const std::string& name, i32 v, i32 min, i32 max, bool hideTune = false) {
    return AddUCIOption(name, v, min, max, std::max(0.5, (max - min) / 20.0), hideTune);
}

inline TunableOption& AddUCIOption(const std::string& name, i32 v) {
    auto min = static_cast<i32>(std::round(v * (1 - 0.45)));
    auto max = static_cast<i32>(std::round(v * (1 + 0.45)));
    return AddUCIOption(name, v, min, max);
}


//  Regular option
#define UCI_OPTION(Name, Default) \
    inline TunableOption& Name = AddUCIOption(#Name, Default);

//  Set min/max manually
#define UCI_OPTION_CUSTOM(Name, Default, Min, Max) \
    inline TunableOption& Name = AddUCIOption(#Name, Default, Min, Max);

//  Option that isn't tunable, i.e. threads/hash
#define UCI_OPTION_SPECIAL(Name, Default, Min, Max) \
    inline TunableOption& Name = AddUCIOption(#Name, Default, Min, Max, true);

//  Spin option, i.e. UCI_Chess960
#define UCI_OPTION_SPIN(Name, Default) \
    inline TunableOption& Name = AddUCIOption(#Name, Default, false, true, 1, true);


#endif // !TUNABLE_H
