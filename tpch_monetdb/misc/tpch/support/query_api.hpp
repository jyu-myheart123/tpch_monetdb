#pragma once


struct Engine;


void query(Engine*);

struct QueryApi {
    void (*query)(Engine*);
};
