//
//  cli_version_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/config.h"
#include "coreml_test_config.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/wait.h>

namespace {

/**
 * @brief Run a shell command and capture its standard output.
 *
 * @param command Shell command to execute.
 * @param exit_code Receives the process exit status.
 * @return Captured standard output.
 */
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

bool test_coreml_metadata_loads() {
    beatit::BeatitConfig config;
    beatit::tests::apply_beatthis_coreml_test_config(config);
    config.model_path = beatit::tests::resolve_beatthis_coreml_model_path();

    if (config.model_path.empty()) {
        std::cerr << "CLI version test failed: could not locate BeatThis model package.\n";
        return false;
    }

    const beatit::CoreMLMetadata metadata = beatit::load_coreml_metadata(config);
    if (metadata.short_description != "Beat This! (small0) - Beat Tracking") {
        std::cerr << "CLI version test failed: unexpected short description '"
                  << metadata.short_description << "'.\n";
        return false;
    }

    if (metadata.version != "small0") {
        std::cerr << "CLI version test failed: unexpected model version '" << metadata.version
                  << "'.\n";
        return false;
    }

    return true;
}

bool test_cli_version_output() {
#if !defined(BEATIT_TEST_CLI_PATH) || !defined(BEATIT_TEST_DATA_DIR)
    std::cerr << "CLI version test failed: BEATIT_TEST_CLI_PATH is not defined.\n";
    return false;
#else
    int exit_code = -1;
    const std::string command = std::string("cd \"") + BEATIT_TEST_DATA_DIR +
        "\" && \"" + BEATIT_TEST_CLI_PATH + "\" --version";
    const std::string output = run_command(command, &exit_code);
    if (exit_code != 0) {
        std::cerr << "CLI version test failed: `beatit --version` exited with "
                  << exit_code << ".\n";
        return false;
    }

    if (output.find("BeatIt v") != 0) {
        std::cerr << "CLI version test failed: unexpected banner '" << output << "'.\n";
        return false;
    }

    if (output.find("Beat This! small0") == std::string::npos) {
        std::cerr << "CLI version test failed: model label missing from banner '" << output
                  << "'.\n";
        return false;
    }

    return true;
#endif
}

} // namespace

int main() {
    if (!test_coreml_metadata_loads()) {
        return 1;
    }

    if (!test_cli_version_output()) {
        return 1;
    }

    return 0;
}
