//
//  cli_model_info_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <string>
#include <sys/wait.h>

namespace {

std::string run_command(const std::string& command, int* exit_code) {
    std::string output;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        if (exit_code) {
            *exit_code = -1;
        }
        return output;
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }

    const int status = pclose(pipe);
    if (exit_code) {
        *exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : status;
    }
    return output;
}

bool validate_coreml_model_info() {
    int exit_code = -1;
    const std::string command = std::string("cd \"") + BEATIT_TEST_DATA_DIR +
        "\" && \"" + BEATIT_TEST_CLI_PATH + "\" --model-info";
    const std::string output = run_command(command, &exit_code);
    if (exit_code != 0) {
        std::cerr << "CLI model-info test failed: CoreML command exited with "
                  << exit_code << ".\n";
        return false;
    }

    if (output.find("CoreML metadata:\n") != 0) {
        std::cerr << "CLI model-info test failed: unexpected CoreML output '" << output
                  << "'.\n";
        return false;
    }

    if (output.find("Description: Beat This! (small0) - Beat Tracking") ==
        std::string::npos) {
        std::cerr << "CLI model-info test failed: CoreML description missing from '"
                  << output << "'.\n";
        return false;
    }

    return true;
}

bool validate_torch_model_info() {
    int exit_code = -1;
    const std::string command = std::string("cd \"") + BEATIT_TEST_DATA_DIR +
        "\" && \"" + BEATIT_TEST_CLI_PATH +
        "\" --model-info --backend torch --torch-model models/beatthis.pt";
    const std::string output = run_command(command, &exit_code);
    if (exit_code != 0) {
        std::cerr << "CLI model-info test failed: Torch command exited with "
                  << exit_code << ".\n";
        return false;
    }

    if (output.find("Torch model:\n") != 0) {
        std::cerr << "CLI model-info test failed: unexpected Torch output '" << output
                  << "'.\n";
        return false;
    }

    if (output.find("Path: models/beatthis.pt") == std::string::npos) {
        std::cerr << "CLI model-info test failed: Torch path missing from '" << output
                  << "'.\n";
        return false;
    }

    if (output.find("Description: BeatThis") == std::string::npos) {
        std::cerr << "CLI model-info test failed: Torch description missing from '"
                  << output << "'.\n";
        return false;
    }

    if (output.find("Configured device: mps") == std::string::npos) {
        std::cerr << "CLI model-info test failed: Torch configured device missing from '"
                  << output << "'.\n";
        return false;
    }

    return true;
}

} // namespace

int main() {
    if (!validate_coreml_model_info()) {
        return 1;
    }

    if (!validate_torch_model_info()) {
        return 1;
    }

    return 0;
}
