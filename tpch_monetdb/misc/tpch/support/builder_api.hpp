#pragma once

struct RawData;
struct Engine;

Engine* build(RawData*);


struct BuilderApi {
    Engine* (*build)(RawData*);
};
