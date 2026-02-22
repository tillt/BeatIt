//
//  logging.hpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace beatit {

struct CoreMLConfig;

/// @brief Global log verbosity for BeatIt output streams.
enum class LogVerbosity { Error = 0, Warn = 1, Info = 2, Debug = 3 };

/// @brief Set the current global BeatIt log verbosity.
void set_log_verbosity(LogVerbosity level);

/// @brief Get the current global BeatIt log verbosity.
LogVerbosity get_log_verbosity();

/// @brief Configure BeatIt log verbosity from config flags.
void set_log_verbosity_from_config(const CoreMLConfig& config);

/// @brief Hex-preview helper for debug dumps of short byte prefixes.
inline constexpr std::size_t kHexPreviewBytes = 8;

/// @brief Format the first `max_len` bytes as a compact hex string.
inline std::string hex_prefix(const std::vector<std::uint8_t>& data,
                              std::size_t max_len = kHexPreviewBytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    const std::size_t limit = std::min(max_len, data.size());
    for (std::size_t i = 0; i < limit; ++i) {
        out << std::setw(2) << static_cast<unsigned int>(data[i]);
        if (i + 1 != limit) {
            out << ' ';
        }
    }
    return out.str();
}

} // namespace beatit

inline constexpr beatit::LogVerbosity beatit_severity_for_tag(std::string_view tag) {
    if (tag == "error") {
        return beatit::LogVerbosity::Error;
    }
    if (tag == "warn" || tag == "warning") {
        return beatit::LogVerbosity::Warn;
    }
    if (tag == "info") {
        return beatit::LogVerbosity::Info;
    }
    return beatit::LogVerbosity::Debug;
}

inline bool beatit_should_log(const char* level) {
    const auto current = beatit::get_log_verbosity();
    const auto severity = beatit_severity_for_tag(level ? level : "");
    return static_cast<int>(severity) <= static_cast<int>(current);
}

inline void beatit_log_impl(const char* level,
                            const std::string& message,
                            const char* file,
                            int line,
                            const char* func) {
    const std::string label = level ? level : "";
    if (label == "error") {
        std::cerr << "[BeatIt][" << label << "][" << file << ":" << line
                  << " " << func << "] " << message << "\n";
        return;
    }

    std::cerr << "[BeatIt][" << label << "] " << message << "\n";
}

#define BEATIT_LOG(level, message)                                               \
    do {                                                                         \
        if (beatit_should_log(level)) {                                          \
            std::ostringstream _beatit_log_stream;                               \
            _beatit_log_stream << message;                                       \
            beatit_log_impl(level,                                               \
                            _beatit_log_stream.str(),                            \
                            __FILE__,                                            \
                            __LINE__,                                            \
                            __func__);                                           \
        }                                                                        \
    } while (0)

#define BEATIT_LOG_ERROR(message) BEATIT_LOG("error", message)
#define BEATIT_LOG_WARN(message) BEATIT_LOG("warn", message)
#define BEATIT_LOG_INFO(message) BEATIT_LOG("info", message)
#define BEATIT_LOG_DEBUG(message) BEATIT_LOG("debug", message)
