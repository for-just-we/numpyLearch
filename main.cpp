//
// Created by prophe cheng on 2022/12/1.
//
#include <cassert>
#include "learchModel.h"
using namespace std;

int main(int argc, char** argv) {
    assert(argc == 2);
    string path(argv[1]);
    PolicyFeedForwardNp model(path);
    vector<double> vec(47, 1);
    model.forward(vec);

    return 0;
}