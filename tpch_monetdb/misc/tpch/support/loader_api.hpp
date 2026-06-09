#pragma once

#include <string>

struct RawData;

RawData* load(std::string);

struct LoaderApi {
    RawData* (*load)(std::string);
};
