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

/// @brief Logging level policy for BeatIt.
///
/// Usage contract:
/// - `Error`: hard failures that prevent the requested operation.
/// - `Warn`: degraded behavior, fallback, invalid/missing runtime inputs, or
///   any suspicious condition users should see in non-verbose mode.
/// - `Info`: high-level lifecycle/profiling summaries (e.g. timing lines).
/// - `Debug`: deep diagnostics, probes, candidate scores, and internal traces.
///
/// Important:
/// - `Warn` and `Error` logs must never be additionally gated by local flags
///   such as ad-hoc local booleans; global logger level controls visibility.
/// - `Debug` may be frequent and verbose by design.
enum class LogVerbosity {
    /// @brief Hard failure; requested operation cannot be completed.
    Error = 0,
    /// @brief Recoverable issue, fallback, or suspicious condition.
    Warn = 1,
    /// @brief Operational summary and profiling information.
    Info = 2,
    /// @brief Detailed internal diagnostics and trace data.
    Debug = 3
};

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

inline void beatit_log_multiline_impl(const char* level,
                                      const std::string& message,
                                      const char* file,
                                      int line,
                                      const char* func) {
    if (message.empty()) {
        beatit_log_impl(level, message, file, line, func);
        return;
    }

    std::size_t start = 0;
    while (start <= message.size()) {
        const std::size_t end = message.find('\n', start);
        const std::size_t len =
            (end == std::string::npos) ? (message.size() - start) : (end - start);
        const std::string line_msg = message.substr(start, len);
        if (!line_msg.empty()) {
            beatit_log_impl(level, line_msg, file, line, func);
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
}

namespace beatit {

/// @brief Stream-style logger adapter for building log messages across many statements.
class LogStream {
public:
    LogStream(const char* level, const char* file, int line, const char* func)
        : level_(level),
          file_(file),
          line_(line),
          func_(func),
          enabled_(beatit_should_log(level)) {}

    template <typename T>
    LogStream& operator<<(const T& value) {
        if (enabled_) {
            stream_ << value;
        }
        return *this;
    }

    LogStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        if (enabled_) {
            manip(stream_);
        }
        return *this;
    }

    ~LogStream() {
        if (!enabled_) {
            return;
        }
        beatit_log_multiline_impl(level_, stream_.str(), file_, line_, func_);
    }

private:
    const char* level_ = nullptr;
    const char* file_ = nullptr;
    int line_ = 0;
    const char* func_ = nullptr;
    bool enabled_ = false;
    std::ostringstream stream_;
};

} // namespace beatit

#define BEATIT_LOG(level, message)                                               \
    do {                                                                         \
        if (beatit_should_log(level)) {                                          \
            std::ostringstream _beatit_log_stream;                               \
            _beatit_log_stream << message;                                       \
            beatit_log_multiline_impl(level,                                     \
                                      _beatit_log_stream.str(),                  \
                                      __FILE__,                                  \
                                      __LINE__,                                  \
                                      __func__);                                 \
        }                                                                        \
    } while (0)

#define BEATIT_LOG_STREAM(level) ::beatit::LogStream(level, __FILE__, __LINE__, __func__)
#define BEATIT_LOG_ERROR_STREAM() BEATIT_LOG_STREAM("error")
#define BEATIT_LOG_WARN_STREAM() BEATIT_LOG_STREAM("warn")
#define BEATIT_LOG_INFO_STREAM() BEATIT_LOG_STREAM("info")
#define BEATIT_LOG_DEBUG_STREAM() BEATIT_LOG_STREAM("debug")

#define BEATIT_LOG_ERROR(message) BEATIT_LOG("error", message)
#define BEATIT_LOG_WARN(message) BEATIT_LOG("warn", message)
#define BEATIT_LOG_INFO(message) BEATIT_LOG("info", message)
#define BEATIT_LOG_DEBUG(message) BEATIT_LOG("debug", message)
