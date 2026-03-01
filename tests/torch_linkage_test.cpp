//
//  torch_linkage_test.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string run_command(const std::string& command, int* status) {
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        if (status) {
            *status = -1;
        }
        return {};
    }

    std::string output;
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }

    const int command_status = pclose(pipe);
    if (status) {
        *status = command_status;
    }
    return output;
}

bool contains_torch_link(const std::string& output) {
    return output.find("libtorch.dylib") != std::string::npos ||
           output.find("libtorch_cpu.dylib") != std::string::npos ||
           output.find("libc10.dylib") != std::string::npos;
}

} // namespace

int main() {
    int status = 0;
    const std::string binary_output =
        run_command(std::string("otool -L \"") + BEATIT_TEST_MAIN_BINARY + "\"", &status);
    if (status != 0) {
        std::cerr << "Torch linkage test failed: could not inspect main binary.\n";
        return 1;
    }

    if (contains_torch_link(binary_output)) {
        std::cerr << "Torch linkage test failed: main binary still links Torch.\n";
        return 1;
    }

#ifdef BEATIT_TEST_TORCH_PLUGIN
    const std::string plugin_output =
        run_command(std::string("otool -L \"") + BEATIT_TEST_TORCH_PLUGIN + "\"", &status);
    if (status != 0) {
        std::cerr << "Torch linkage test failed: could not inspect Torch plugin.\n";
        return 1;
    }

    if (!contains_torch_link(plugin_output)) {
        std::cerr << "Torch linkage test failed: Torch plugin does not link Torch.\n";
        return 1;
    }
#endif

    return 0;
}
