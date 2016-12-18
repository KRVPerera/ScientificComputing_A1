//
// Created by krv on 12/16/16.
//

#include<iostream>
#include <A1Config.h>
#include "Util.h"

using namespace std;

int main(int argc, char **argv) {
    ios_base::sync_with_stdio(0);
    cout << "Hello from Q3" << endl;
    vector<float> mm;
    mm.push_back(5.0);
    mm.push_back(5.0);
    mm.push_back(5.0);
    mm.push_back(5.0);
    cout << Util::Mean(mm) << endl;
    cout << CS4552_A1_VERSION_MAJOR << endl;
}