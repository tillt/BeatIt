cmake_minimum_required(VERSION 3.20)

if(APPLE)
    # Let Apple Clang manage libc++ implicitly; avoid explicit CMake -lc++ injection.
    set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "")
    set(CMAKE_OBJCXX_STANDARD_LIBRARIES_INIT "")
endif()

project(beatit
    VERSION 0.1.0
    DESCRIPTION "A Minimal BPM/beat tracker using CoreML. Lightweight, permissive."
    LANGUAGES CXX OBJCXX)

set(BEATIT_VERSION_STRING "dev")
set(BEATIT_VERSION_DEV TRUE)
set(BEATIT_GIT_HASH "unknown")
find_program(GIT_EXE git)
if(GIT_EXE)
    execute_process(
        COMMAND ${GIT_EXE} describe --tags --dirty --abbrev=7
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GIT_DESC_RES
        OUTPUT_VARIABLE GIT_DESC_OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(GIT_DESC_RES EQUAL 0 AND NOT GIT_DESC_OUT STREQUAL "")
        set(BEATIT_VERSION_STRING "${GIT_DESC_OUT}")
        if(GIT_DESC_OUT MATCHES "-")
            set(BEATIT_VERSION_DEV TRUE)
        else()
            set(BEATIT_VERSION_DEV FALSE)
        endif()
    endif()

    execute_process(
        COMMAND ${GIT_EXE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GIT_HASH_RES
        OUTPUT_VARIABLE GIT_HASH_OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(GIT_HASH_RES EQUAL 0 AND NOT GIT_HASH_OUT STREQUAL "")
        set(BEATIT_GIT_HASH "${GIT_HASH_OUT}")
    endif()
endif()

if(BEATIT_VERSION_DEV)
    set(BEATIT_VERSION_DISPLAY "${BEATIT_VERSION_STRING}+${BEATIT_GIT_HASH}")
else()
    set(BEATIT_VERSION_DISPLAY "${BEATIT_VERSION_STRING}")
endif()

file(GENERATE
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated/beatit_version.hpp
    CONTENT "//
//  beatit_version.hpp
//

#pragma once

#define BEATIT_VERSION \"${BEATIT_VERSION_STRING}\"
#define BEATIT_VERSION_DISPLAY \"${BEATIT_VERSION_DISPLAY}\"
#define BEATIT_VERSION_IS_DEV ${BEATIT_VERSION_DEV}
#define BEATIT_GIT_HASH \"${BEATIT_GIT_HASH}\"
"
)

if(APPLE AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Avoid duplicate -lc++ on Apple Clang: driver already injects libc++.
    set(CMAKE_OBJCXX_IMPLICIT_LINK_LIBRARIES "")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

option(BEATIT_USE_TORCH "Enable Torch backend" ON)
set(BEATIT_TORCH_ROOT "" CACHE PATH "Path to LibTorch root")

set(BEATIT_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set_property(DIRECTORY APPEND PROPERTY
    ADDITIONAL_CLEAN_FILES "${BEATIT_GENERATED_DIR}")

if(BEATIT_USE_TORCH AND BEATIT_TORCH_ROOT STREQUAL "")
    set(_beatit_torch_candidates
        "/opt/homebrew/opt/pytorch"
        "/usr/local/opt/pytorch"
        "/opt/homebrew/opt/libtorch"
        "/usr/local/opt/libtorch")
    foreach(_prefix IN LISTS _beatit_torch_candidates)
        file(GLOB _beatit_python_torch_candidates
            "${_prefix}/libexec/lib/python*/site-packages/torch")
        list(APPEND _beatit_torch_candidates ${_beatit_python_torch_candidates})
    endforeach()
    foreach(_cand IN LISTS _beatit_torch_candidates)
        if(EXISTS "${_cand}/share/cmake/Torch" OR
           EXISTS "${_cand}/torch/share/cmake/Torch" OR
           EXISTS "${_cand}/libtorch/share/cmake/Torch")
            set(BEATIT_TORCH_ROOT "${_cand}" CACHE PATH "Path to LibTorch root" FORCE)
            break()
        endif()
    endforeach()
endif()

set(BEATIT_LIB_SOURCES
    src/beatit/analysis/core.cpp
    src/beatit/analysis/bpm_beats.cpp
    src/beatit/analysis/bpm_activation.cpp
    src/beatit/analysis/backend.cpp
    src/beatit/analysis/torch_loader.mm
    src/beatit/audio/dsp.cpp
    src/beatit/audio/features.cpp
    src/beatit/audio/wav.cpp
    src/beatit/runtime/logging.cpp
    src/beatit/runtime/version.cpp
    src/beatit/inference/coreml/model_utils.mm
    src/beatit/inference/coreml/metadata.mm
    src/beatit/inference/backend_base.cpp
    src/beatit/inference/coreml_loader.mm
    src/beatit/inference/coreml_plugin.mm
    src/beatit/inference/torch_plugin.mm
    src/beatit/post/pipeline.cpp
    src/beatit/post/helpers.cpp
    src/beatit/post/dbn_run.cpp
    src/beatit/post/dbn_apply.cpp
    src/beatit/post/grid_anchor.cpp
    src/beatit/post/grid_phase_select.cpp
    src/beatit/post/grid_synthesize.cpp
    src/beatit/post/grid_tempo_decision.cpp
    src/beatit/post/logits.cpp
    src/beatit/post/grid_fill.cpp
    src/beatit/post/result_ops.cpp
    src/beatit/post/tempo_fit.cpp
    src/beatit/post/window_select.cpp
    src/beatit/post/window.cpp
    src/beatit/coreml/preset.cpp
    src/beatit/dbn/calmdad.cpp
    src/beatit/dbn/calmdad_sparse.cpp
    src/beatit/dbn/beatit.cpp
    src/beatit/refiner/core.cpp
    src/beatit/refiner/constant.cpp
    src/beatit/refiner/constant_phase.cpp
    src/beatit/refiner/constant_support.cpp
    src/beatit/refiner/constant_regions.cpp
    src/beatit/inference/window_merge.cpp
    src/beatit/stream/finalize.cpp
    src/beatit/inference/backend_coreml_loader.mm
    src/beatit/inference/backend_torch_loader.mm
    src/beatit/sparse/probe.cpp
    src/beatit/sparse/probe_pick.cpp
    src/beatit/sparse/probe_pick_helpers.cpp
    src/beatit/sparse/probe_score.cpp
    src/beatit/sparse/usability_scan.cpp
    src/beatit/sparse/phase_metrics.cpp
    src/beatit/sparse/interior_select.cpp
    src/beatit/sparse/refinement.cpp
    src/beatit/sparse/edge.cpp
    src/beatit/sparse/edge_adjust.cpp
    src/beatit/sparse/edge_metrics.cpp
    src/beatit/sparse/edge_phase.cpp
    src/beatit/sparse/waveform.cpp
    src/beatit/stream/core.cpp
)

add_library(beatit_objects OBJECT ${BEATIT_LIB_SOURCES})
set_target_properties(beatit_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(beatit_objects
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}/generated
)

add_library(beatit_lib STATIC $<TARGET_OBJECTS:beatit_objects>)

target_include_directories(beatit_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}/generated
)

function(beatit_link_torch target)
    if(NOT BEATIT_USE_TORCH)
        return()
    endif()
    if(BEATIT_TORCH_ROOT STREQUAL "")
        message(FATAL_ERROR "BEATIT_TORCH_ROOT must be set when BEATIT_USE_TORCH=ON")
    endif()
    set(_beatit_torch_cmake_dir "")
    set(_beatit_torch_lib_dir "")
    if(EXISTS "${BEATIT_TORCH_ROOT}/share/cmake/Torch")
        set(_beatit_torch_cmake_dir "${BEATIT_TORCH_ROOT}/share/cmake/Torch")
        set(_beatit_torch_lib_dir "${BEATIT_TORCH_ROOT}/lib")
    elseif(EXISTS "${BEATIT_TORCH_ROOT}/torch/share/cmake/Torch")
        set(_beatit_torch_cmake_dir "${BEATIT_TORCH_ROOT}/torch/share/cmake/Torch")
        set(_beatit_torch_lib_dir "${BEATIT_TORCH_ROOT}/torch/lib")
    elseif(EXISTS "${BEATIT_TORCH_ROOT}/libtorch/share/cmake/Torch")
        set(_beatit_torch_cmake_dir "${BEATIT_TORCH_ROOT}/libtorch/share/cmake/Torch")
        set(_beatit_torch_lib_dir "${BEATIT_TORCH_ROOT}/libtorch/lib")
    else()
        message(FATAL_ERROR
            "Could not locate Torch CMake config under BEATIT_TORCH_ROOT='${BEATIT_TORCH_ROOT}'. "
            "Checked: share/cmake/Torch, torch/share/cmake/Torch, libtorch/share/cmake/Torch")
    endif()
    set(Torch_DIR "${_beatit_torch_cmake_dir}")
    find_package(Torch REQUIRED)
    target_include_directories(${target} SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    target_compile_definitions(${target} PRIVATE BEATIT_USE_TORCH=1)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target} PRIVATE -Wno-gnu-zero-variadic-macro-arguments)
    endif()

    get_target_property(_beatit_target_type ${target} TYPE)
    if(_beatit_target_type STREQUAL "OBJECT_LIBRARY")
        return()
    endif()

    set(BEATIT_TORCH_LIBS
        torch
        torch_cpu
        c10
    )
    find_library(TORCH_GLOBAL_DEPS_LIB
        NAMES torch_global_deps
        PATHS "${_beatit_torch_lib_dir}"
        NO_DEFAULT_PATH
    )
    if(TORCH_GLOBAL_DEPS_LIB)
        list(APPEND BEATIT_TORCH_LIBS "${TORCH_GLOBAL_DEPS_LIB}")
    endif()
    target_link_libraries(${target} PRIVATE ${BEATIT_TORCH_LIBS})
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH "${_beatit_torch_lib_dir}"
    )
endfunction()

add_executable(beatit
    src/cli/main.mm
)

target_include_directories(beatit
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(beatit
    PRIVATE
        beatit_lib
)

add_library(beatit_coreml_plugin_support STATIC
    src/beatit/runtime/logging.cpp
    src/beatit/audio/dsp.cpp
    src/beatit/audio/features.cpp
    src/beatit/inference/backend_base.cpp
    src/beatit/inference/coreml/model_utils.mm
)

target_include_directories(beatit_coreml_plugin_support
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}/generated
)

add_library(beatit_backend_coreml_plugin SHARED
    src/beatit/inference/backend_coreml.cpp
    src/beatit/inference/coreml/inference.mm
)

target_include_directories(beatit_backend_coreml_plugin
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_compile_definitions(beatit_backend_coreml_plugin PRIVATE BEATIT_COREML_PLUGIN_BUILD=1)

target_link_libraries(beatit_backend_coreml_plugin
    PRIVATE
        beatit_coreml_plugin_support
)

set_target_properties(beatit_backend_coreml_plugin PROPERTIES
    OUTPUT_NAME "beatit_backend_coreml"
)

add_dependencies(beatit beatit_backend_coreml_plugin)
add_custom_command(TARGET beatit POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:beatit>/plugins"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:beatit_backend_coreml_plugin>"
        "$<TARGET_FILE_DIR:beatit>/plugins/$<TARGET_FILE_NAME:beatit_backend_coreml_plugin>"
    COMMENT "Copying CoreML backend plugin next to beatit CLI"
)

if(BEATIT_USE_TORCH)
    add_library(beatit_torch_plugin_support STATIC
        src/beatit/runtime/logging.cpp
        src/beatit/audio/dsp.cpp
        src/beatit/audio/features.cpp
        src/beatit/inference/backend_base.cpp
    )

    target_include_directories(beatit_torch_plugin_support
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/generated
    )

    add_library(beatit_backend_torch_plugin SHARED
        src/beatit/analysis/torch.cpp
        src/beatit/inference/backend_torch.cpp
        src/beatit/audio/mel_torch.cpp
    )

    target_include_directories(beatit_backend_torch_plugin
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/generated
    )

    beatit_link_torch(beatit_backend_torch_plugin)

    target_link_libraries(beatit_backend_torch_plugin
        PRIVATE
            beatit_torch_plugin_support
    )

    set_target_properties(beatit_backend_torch_plugin PROPERTIES
        OUTPUT_NAME "beatit_backend_torch"
    )

    add_dependencies(beatit beatit_backend_torch_plugin)
    add_custom_command(TARGET beatit POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:beatit>/plugins"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:beatit_backend_torch_plugin>"
            "$<TARGET_FILE_DIR:beatit>/plugins/$<TARGET_FILE_NAME:beatit_backend_torch_plugin>"
        COMMENT "Copying Torch backend plugin next to beatit CLI"
    )
endif()

if(APPLE)
    target_compile_options(beatit_objects PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_compile_options(beatit PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_lib PRIVATE "-framework Accelerate" "-framework CoreML" "-framework Foundation")
    target_link_libraries(beatit_coreml_plugin_support PRIVATE "-framework Accelerate" "-framework CoreML" "-framework Foundation")
    target_link_libraries(beatit_backend_coreml_plugin PRIVATE "-framework Accelerate" "-framework CoreML" "-framework Foundation")
    target_link_libraries(beatit PRIVATE "-framework AVFoundation" "-framework Foundation" "-framework CoreMedia" "-framework AudioToolbox")
    if(TARGET beatit_torch_plugin_support)
        target_link_libraries(beatit_torch_plugin_support PRIVATE "-framework Accelerate")
    endif()
    if(TARGET beatit_backend_torch_plugin)
        target_link_libraries(beatit_backend_torch_plugin PRIVATE "-framework Accelerate")
    endif()
endif()

add_executable(beatit_tool_probe_runner
    tools/probe_runner.mm
)

target_include_directories(beatit_tool_probe_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(beatit_tool_probe_runner PRIVATE
    beatit_lib
)

add_executable(beatit_tool_alignment_inspector
    tools/alignment_inspector.mm
)

add_executable(beatit_tool_cache_benchmark
    tools/cache_benchmark.mm
)

target_include_directories(beatit_tool_alignment_inspector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(beatit_tool_cache_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(beatit_tool_alignment_inspector PRIVATE
    beatit_lib
)

target_link_libraries(beatit_tool_cache_benchmark PRIVATE
    beatit_lib
)

if(APPLE)
    target_compile_options(beatit_tool_probe_runner PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_compile_options(beatit_tool_alignment_inspector PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_compile_options(beatit_tool_cache_benchmark PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_tool_probe_runner PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
    target_link_libraries(beatit_tool_alignment_inspector PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
    target_link_libraries(beatit_tool_cache_benchmark PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

if(APPLE)
    set(BEATIT_PUBLIC_HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/beatit/analysis.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/beatit/config.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/beatit/coreml_preset.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/beatit/logging.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/beatit/refiner.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/beatit/stream.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/beatit/version.h")

    set(BEATIT_FRAMEWORK_SOURCES
        $<TARGET_OBJECTS:beatit_objects>
        ${BEATIT_PUBLIC_HEADERS}
    )

    add_library(beatit_framework SHARED ${BEATIT_FRAMEWORK_SOURCES})

    target_include_directories(beatit_framework
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/generated
    )

    set_target_properties(beatit_framework PROPERTIES
        FRAMEWORK TRUE
        FRAMEWORK_VERSION A
        OUTPUT_NAME "BeatIt"
        MACOSX_FRAMEWORK_IDENTIFIER "com.tilltoenshoff.beatit"
        PUBLIC_HEADER "${BEATIT_PUBLIC_HEADERS}"
    )

    foreach(header_file IN LISTS BEATIT_PUBLIC_HEADERS)
        set_source_files_properties("${header_file}" PROPERTIES
            MACOSX_PACKAGE_LOCATION "Headers/beatit"
        )
    endforeach()

    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/BeatThis_small0.mlpackage")
        message(FATAL_ERROR
            "Missing required bundled BeatThis CoreML package: "
            "${CMAKE_CURRENT_SOURCE_DIR}/models/BeatThis_small0.mlpackage")
    endif()

    add_custom_command(TARGET beatit_framework POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -rf
            "$<TARGET_FILE_DIR:beatit_framework>/Resources/beatit.mlmodelc"
        COMMAND ${CMAKE_COMMAND} -E rm -rf
            "$<TARGET_FILE_DIR:beatit_framework>/Resources/BeatThis_final0.mlpackage"
        COMMAND ${CMAKE_COMMAND} -E rm -rf
            "$<TARGET_FILE_DIR:beatit_framework>/Resources/BeatThis_small0.mlpackage"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/models/BeatThis_small0.mlpackage"
            "$<TARGET_FILE_DIR:beatit_framework>/Resources/models/BeatThis_small0.mlpackage"
        COMMENT "Copying BeatThis small CoreML package into BeatIt.framework resources"
    )

    add_dependencies(beatit_framework beatit_backend_coreml_plugin)
    add_custom_command(TARGET beatit_framework POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:beatit_framework>/Resources/plugins"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:beatit_backend_coreml_plugin>"
            "$<TARGET_FILE_DIR:beatit_framework>/Resources/plugins/$<TARGET_FILE_NAME:beatit_backend_coreml_plugin>"
        COMMENT "Copying CoreML backend plugin into BeatIt.framework resources"
    )

    if(BEATIT_USE_TORCH)
        add_dependencies(beatit_framework beatit_backend_torch_plugin)
        add_custom_command(TARGET beatit_framework POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:beatit_framework>/Resources/plugins"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:beatit_backend_torch_plugin>"
                "$<TARGET_FILE_DIR:beatit_framework>/Resources/plugins/$<TARGET_FILE_NAME:beatit_backend_torch_plugin>"
            COMMENT "Copying Torch backend plugin into BeatIt.framework resources"
        )
    endif()

    if(BEATIT_USE_TORCH AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/beatthis.pt")
        add_custom_command(TARGET beatit_framework POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/models/beatthis.pt"
                "$<TARGET_FILE_DIR:beatit_framework>/Resources/beatthis.pt"
            COMMENT "Copying BeatThis Torch model into BeatIt.framework resources"
        )
    endif()

    set(BEATIT_FRAMEWORK_HEADER_ROOT "$<TARGET_FILE_DIR:beatit_framework>/Headers")
    add_custom_command(TARGET beatit_framework POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -rf
            "${BEATIT_FRAMEWORK_HEADER_ROOT}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${BEATIT_FRAMEWORK_HEADER_ROOT}/beatit"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${BEATIT_PUBLIC_HEADERS}
            "${BEATIT_FRAMEWORK_HEADER_ROOT}/beatit"
        COMMENT "Staging public BeatIt headers into Headers/beatit only"
    )

    target_compile_options(beatit_framework PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_framework PRIVATE "-framework Accelerate" "-framework CoreML" "-framework Foundation")
endif()

enable_testing()

add_executable(beatit_dbn_tests
    tests/dbn_decoder_test.cpp
)

target_link_libraries(beatit_dbn_tests
    PRIVATE
        beatit_lib
)

if(APPLE)
    target_compile_options(beatit_dbn_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
endif()

add_test(NAME beatit_dbn_tests COMMAND beatit_dbn_tests)

add_executable(beatit_dbn_grid_stage_tests
    tests/dbn_grid_stage_test.cpp
)

target_link_libraries(beatit_dbn_grid_stage_tests
    PRIVATE
        beatit_lib
)

target_include_directories(beatit_dbn_grid_stage_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/beatit
)

if(APPLE)
    target_compile_options(beatit_dbn_grid_stage_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
endif()

add_test(NAME beatit_dbn_grid_stage_tests COMMAND beatit_dbn_grid_stage_tests)

add_executable(beatit_sparse_usability_scan_tests
    tests/sparse_usability_scan_test.cpp
)

target_link_libraries(beatit_sparse_usability_scan_tests
    PRIVATE
        beatit_lib
)

target_include_directories(beatit_sparse_usability_scan_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/beatit
)

if(APPLE)
    target_compile_options(beatit_sparse_usability_scan_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
endif()

add_test(NAME beatit_sparse_usability_scan_tests COMMAND beatit_sparse_usability_scan_tests)

add_executable(beatit_coreml_plugin_load_tests
    tests/coreml_plugin_load_test.mm
)

target_link_libraries(beatit_coreml_plugin_load_tests
    PRIVATE
        beatit_lib
)

target_include_directories(beatit_coreml_plugin_load_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_dependencies(beatit_coreml_plugin_load_tests beatit_backend_coreml_plugin)

if(APPLE)
    target_compile_options(beatit_coreml_plugin_load_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
endif()

add_test(NAME beatit_coreml_plugin_load_tests COMMAND beatit_coreml_plugin_load_tests)

add_executable(beatit_coreml_compile_cache_tests
    tests/coreml_compile_cache_test.mm
)

target_link_libraries(beatit_coreml_compile_cache_tests
    PRIVATE
        beatit_lib
        "-framework CoreML"
        "-framework Foundation"
)

target_include_directories(beatit_coreml_compile_cache_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/beatit
)

target_compile_definitions(beatit_coreml_compile_cache_tests
    PRIVATE
        BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(APPLE)
    target_compile_options(beatit_coreml_compile_cache_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
endif()

add_test(NAME beatit_coreml_compile_cache_tests COMMAND beatit_coreml_compile_cache_tests)
set_tests_properties(beatit_coreml_compile_cache_tests PROPERTIES
    SKIP_RETURN_CODE 77
)

add_executable(beatit_coreml_metadata_tests
    tests/coreml_metadata_test.mm
)

target_link_libraries(beatit_coreml_metadata_tests
    PRIVATE
        beatit_lib
        "-framework CoreML"
        "-framework Foundation"
)

target_include_directories(beatit_coreml_metadata_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/beatit
)

target_compile_definitions(beatit_coreml_metadata_tests
    PRIVATE
        BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(APPLE)
    target_compile_options(beatit_coreml_metadata_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
endif()

add_test(NAME beatit_coreml_metadata_tests COMMAND beatit_coreml_metadata_tests)
set_tests_properties(beatit_coreml_metadata_tests PROPERTIES
    SKIP_RETURN_CODE 77
)

add_executable(beatit_cli_version_tests
    tests/cli_version_test.mm
)

target_link_libraries(beatit_cli_version_tests
    PRIVATE
        beatit_lib
        "-framework Foundation"
)

target_include_directories(beatit_cli_version_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/beatit
)

target_compile_definitions(beatit_cli_version_tests
    PRIVATE
        BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        BEATIT_TEST_CLI_PATH="$<TARGET_FILE:beatit>"
)

add_dependencies(beatit_cli_version_tests beatit)

if(APPLE)
    target_compile_options(beatit_cli_version_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
endif()

add_test(NAME beatit_cli_version_tests COMMAND beatit_cli_version_tests)
set_tests_properties(beatit_cli_version_tests PROPERTIES
    SKIP_RETURN_CODE 77
)
set_tests_properties(beatit_cli_version_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(beatit_cli_model_info_tests
    tests/cli_model_info_test.mm
)

target_link_libraries(beatit_cli_model_info_tests
    PRIVATE
        beatit_lib
        "-framework Foundation"
)

target_include_directories(beatit_cli_model_info_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/beatit
)

target_compile_definitions(beatit_cli_model_info_tests
    PRIVATE
        BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        BEATIT_TEST_CLI_PATH="$<TARGET_FILE:beatit>"
)

add_dependencies(beatit_cli_model_info_tests beatit)

if(APPLE)
    target_compile_options(beatit_cli_model_info_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
endif()

add_test(NAME beatit_cli_model_info_tests COMMAND beatit_cli_model_info_tests)
set_tests_properties(beatit_cli_model_info_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(beatit_sparse_interior_select_tests
    tests/sparse_interior_select_test.cpp
)

target_link_libraries(beatit_sparse_interior_select_tests
    PRIVATE
        beatit_lib
)

target_include_directories(beatit_sparse_interior_select_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/beatit
)

if(APPLE)
    target_compile_options(beatit_sparse_interior_select_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
endif()

add_test(NAME beatit_sparse_interior_select_tests COMMAND beatit_sparse_interior_select_tests)

add_executable(beatit_sparse_edge_phase_tests
    tests/sparse_edge_phase_test.cpp
)

target_link_libraries(beatit_sparse_edge_phase_tests
    PRIVATE
        beatit_lib
)

target_include_directories(beatit_sparse_edge_phase_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/beatit
)

if(APPLE)
    target_compile_options(beatit_sparse_edge_phase_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
endif()

add_test(NAME beatit_sparse_edge_phase_tests COMMAND beatit_sparse_edge_phase_tests)

add_executable(beatit_sparse_edge_adjust_tests
    tests/sparse_edge_adjust_test.cpp
)

target_link_libraries(beatit_sparse_edge_adjust_tests
    PRIVATE
        beatit_lib
)

target_include_directories(beatit_sparse_edge_adjust_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/beatit
)

if(APPLE)
    target_compile_options(beatit_sparse_edge_adjust_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
endif()

add_test(NAME beatit_sparse_edge_adjust_tests COMMAND beatit_sparse_edge_adjust_tests)

add_executable(beatit_dbn_purelove_tests
    tests/dbn_purelove_test.mm
)

target_link_libraries(beatit_dbn_purelove_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_dbn_purelove_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/BeatThis_small0.mlpackage"
    BEATIT_TEST_AUDIO_PATH="${CMAKE_CURRENT_SOURCE_DIR}/training/purelove.wav"
)

if(APPLE)
    target_compile_options(beatit_dbn_purelove_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_dbn_purelove_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_dbn_purelove_tests COMMAND beatit_dbn_purelove_tests)
set_tests_properties(beatit_dbn_purelove_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_sparse_tempo_grid_consistency_tests
    tests/sparse_tempo_grid_consistency_test.mm
)

target_link_libraries(beatit_sparse_tempo_grid_consistency_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_sparse_tempo_grid_consistency_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/BeatThis_small0.mlpackage"
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    BEATIT_TEST_GENERATED_DIR="${BEATIT_GENERATED_DIR}"
)

if(APPLE)
    target_compile_options(beatit_sparse_tempo_grid_consistency_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_sparse_tempo_grid_consistency_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_sparse_tempo_grid_consistency_tests COMMAND beatit_sparse_tempo_grid_consistency_tests)
set_tests_properties(beatit_sparse_tempo_grid_consistency_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_file_drift_probe
    tests/file_drift_probe_test.mm
)

target_link_libraries(beatit_file_drift_probe
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_file_drift_probe PRIVATE
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(APPLE)
    target_compile_options(beatit_file_drift_probe PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_file_drift_probe PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_executable(beatit_manucho_window_alignment_sparse_tests
    tests/manucho_window_alignment_test.mm
)

target_link_libraries(beatit_manucho_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_manucho_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/BeatThis_small0.mlpackage"
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    BEATIT_TEST_SPARSE_PROBE_MODE=1
)

if(APPLE)
    target_compile_options(beatit_manucho_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_manucho_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_manucho_window_alignment_sparse_tests COMMAND beatit_manucho_window_alignment_sparse_tests)
set_tests_properties(beatit_manucho_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_enigma_window_alignment_sparse_tests
    tests/enigma_window_alignment_test.mm
)

target_link_libraries(beatit_enigma_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_enigma_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/BeatThis_small0.mlpackage"
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    BEATIT_TEST_SPARSE_PROBE_MODE=1
)

if(APPLE)
    target_compile_options(beatit_enigma_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_enigma_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_enigma_window_alignment_sparse_tests COMMAND beatit_enigma_window_alignment_sparse_tests)
set_tests_properties(beatit_enigma_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_vamp_window_alignment_sparse_tests
    tests/vamp_window_alignment_test.mm
)

target_link_libraries(beatit_vamp_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_vamp_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/BeatThis_small0.mlpackage"
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    BEATIT_TEST_SPARSE_PROBE_MODE=1
)

if(APPLE)
    target_compile_options(beatit_vamp_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_vamp_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_vamp_window_alignment_sparse_tests COMMAND beatit_vamp_window_alignment_sparse_tests)
set_tests_properties(beatit_vamp_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_eureka_window_alignment_sparse_tests
    tests/eureka_window_alignment_test.mm
)

target_link_libraries(beatit_eureka_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_eureka_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/BeatThis_small0.mlpackage"
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    BEATIT_TEST_SPARSE_PROBE_MODE=1
)

if(APPLE)
    target_compile_options(beatit_eureka_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_eureka_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_eureka_window_alignment_sparse_tests COMMAND beatit_eureka_window_alignment_sparse_tests)
set_tests_properties(beatit_eureka_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_bazaar_window_alignment_sparse_tests
    tests/bazaar_window_alignment_test.mm
)

target_link_libraries(beatit_bazaar_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_bazaar_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/BeatThis_small0.mlpackage"
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    BEATIT_TEST_SPARSE_PROBE_MODE=1
)

if(APPLE)
    target_compile_options(beatit_bazaar_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_bazaar_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_bazaar_window_alignment_sparse_tests COMMAND beatit_bazaar_window_alignment_sparse_tests)
set_tests_properties(beatit_bazaar_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_intelligence_window_alignment_sparse_tests
    tests/intelligence_window_alignment_test.mm
)

target_link_libraries(beatit_intelligence_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_intelligence_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/BeatThis_small0.mlpackage"
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    BEATIT_TEST_SPARSE_PROBE_MODE=1
)

if(APPLE)
    target_compile_options(beatit_intelligence_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_intelligence_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_intelligence_window_alignment_sparse_tests COMMAND beatit_intelligence_window_alignment_sparse_tests)
set_tests_properties(beatit_intelligence_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_best_window_alignment_sparse_tests
    tests/best_window_alignment_test.mm
)

target_link_libraries(beatit_best_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_best_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/BeatThis_small0.mlpackage"
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    BEATIT_TEST_SPARSE_PROBE_MODE=1
)

if(APPLE)
    target_compile_options(beatit_best_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_best_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_best_window_alignment_sparse_tests COMMAND beatit_best_window_alignment_sparse_tests)
set_tests_properties(beatit_best_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_moderat_window_alignment_sparse_tests
    tests/moderat_window_alignment_test.mm
)

target_link_libraries(beatit_moderat_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_moderat_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(APPLE)
    target_compile_options(beatit_moderat_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_moderat_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_moderat_window_alignment_sparse_tests COMMAND beatit_moderat_window_alignment_sparse_tests)
set_tests_properties(beatit_moderat_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_lethal_window_alignment_sparse_tests
    tests/lethal_window_alignment_test.mm
)

target_link_libraries(beatit_lethal_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_lethal_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(APPLE)
    target_compile_options(beatit_lethal_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_lethal_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_lethal_window_alignment_sparse_tests COMMAND beatit_lethal_window_alignment_sparse_tests)
set_tests_properties(beatit_lethal_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_neural_window_alignment_sparse_tests
    tests/neural_window_alignment_test.mm
)

target_link_libraries(beatit_neural_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_neural_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/BeatThis_small0.mlpackage"
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    BEATIT_TEST_SPARSE_PROBE_MODE=1
)

if(APPLE)
    target_compile_options(beatit_neural_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_neural_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_neural_window_alignment_sparse_tests COMMAND beatit_neural_window_alignment_sparse_tests)
set_tests_properties(beatit_neural_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_samerano_window_alignment_sparse_tests
    tests/samerano_window_alignment_test.mm
)

target_link_libraries(beatit_samerano_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_samerano_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(APPLE)
    target_compile_options(beatit_samerano_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_samerano_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_samerano_window_alignment_sparse_tests COMMAND beatit_samerano_window_alignment_sparse_tests)
set_tests_properties(beatit_samerano_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_turbo_window_alignment_sparse_tests
    tests/turbo_window_alignment_test.mm
)

target_link_libraries(beatit_turbo_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_turbo_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(APPLE)
    target_compile_options(beatit_turbo_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_turbo_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_turbo_window_alignment_sparse_tests COMMAND beatit_turbo_window_alignment_sparse_tests)
set_tests_properties(beatit_turbo_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_rhodes_window_alignment_sparse_tests
    tests/rhodes_window_alignment_test.mm
)

target_link_libraries(beatit_rhodes_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_rhodes_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(APPLE)
    target_compile_options(beatit_rhodes_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_rhodes_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_rhodes_window_alignment_sparse_tests COMMAND beatit_rhodes_window_alignment_sparse_tests)
set_tests_properties(beatit_rhodes_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_moving_window_alignment_sparse_tests
    tests/moving_window_alignment_test.mm
)

target_link_libraries(beatit_moving_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_moving_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(APPLE)
    target_compile_options(beatit_moving_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_moving_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_moving_window_alignment_sparse_tests COMMAND beatit_moving_window_alignment_sparse_tests)
set_tests_properties(beatit_moving_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_street_window_alignment_sparse_tests
    tests/street_window_alignment_test.mm
)

target_link_libraries(beatit_street_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_street_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(APPLE)
    target_compile_options(beatit_street_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_street_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_street_window_alignment_sparse_tests COMMAND beatit_street_window_alignment_sparse_tests)
set_tests_properties(beatit_street_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_what_window_alignment_sparse_tests
    tests/what_window_alignment_test.mm
)

target_link_libraries(beatit_what_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_what_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(APPLE)
    target_compile_options(beatit_what_window_alignment_sparse_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_what_window_alignment_sparse_tests PRIVATE "-framework AVFoundation" "-framework CoreML" "-framework Foundation")
endif()

add_test(NAME beatit_what_window_alignment_sparse_tests COMMAND beatit_what_window_alignment_sparse_tests)
set_tests_properties(beatit_what_window_alignment_sparse_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

if(BEATIT_USE_TORCH)
    add_executable(beatit_beatthis_canonical_tests
        tests/beatthis_canonical_test.mm
    )

    target_link_libraries(beatit_beatthis_canonical_tests
        PRIVATE
            beatit_lib
    )

    add_dependencies(beatit_beatthis_canonical_tests beatit_backend_torch_plugin)

    target_compile_definitions(beatit_beatthis_canonical_tests PRIVATE
        BEATIT_TEST_TORCH_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/beatthis.pt"
        BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    if(APPLE)
        target_compile_options(beatit_beatthis_canonical_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
        target_link_libraries(beatit_beatthis_canonical_tests PRIVATE "-framework AVFoundation" "-framework Foundation")
    endif()

    add_test(NAME beatit_beatthis_canonical_tests COMMAND beatit_beatthis_canonical_tests)
    set_tests_properties(beatit_beatthis_canonical_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        SKIP_RETURN_CODE 77
    )

    add_executable(beatit_beatthis_mps_tests
        tests/beatthis_mps_smoke_test.mm
    )

    target_link_libraries(beatit_beatthis_mps_tests
        PRIVATE
            beatit_lib
    )

    beatit_link_torch(beatit_beatthis_mps_tests)
    add_dependencies(beatit_beatthis_mps_tests beatit_backend_torch_plugin)

    target_compile_definitions(beatit_beatthis_mps_tests PRIVATE
        BEATIT_TEST_TORCH_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/beatthis.pt"
        BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    if(APPLE)
        target_compile_options(beatit_beatthis_mps_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
        target_link_libraries(beatit_beatthis_mps_tests PRIVATE "-framework AVFoundation" "-framework Foundation")
    endif()

    add_test(NAME beatit_beatthis_mps_tests COMMAND beatit_beatthis_mps_tests)
    set_tests_properties(beatit_beatthis_mps_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        SKIP_RETURN_CODE 77
    )
endif()

if(APPLE)
    add_executable(beatit_torch_linkage_tests
        tests/torch_linkage_test.cpp
    )

    target_compile_definitions(beatit_torch_linkage_tests PRIVATE
        BEATIT_TEST_MAIN_BINARY="$<TARGET_FILE:beatit>"
    )

    if(BEATIT_USE_TORCH)
        target_compile_definitions(beatit_torch_linkage_tests PRIVATE
            BEATIT_TEST_TORCH_PLUGIN="$<TARGET_FILE:beatit_backend_torch_plugin>"
        )
        add_dependencies(beatit_torch_linkage_tests beatit beatit_backend_torch_plugin)
    else()
        add_dependencies(beatit_torch_linkage_tests beatit)
    endif()

    target_link_libraries(beatit_torch_linkage_tests PRIVATE
        beatit_lib
    )

    add_test(NAME beatit_torch_linkage_tests COMMAND beatit_torch_linkage_tests)
    set_tests_properties(beatit_torch_linkage_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_test(
        NAME beatit_torchscript_model_device_tests
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/torchscript_model_device_test.py
                ${CMAKE_CURRENT_SOURCE_DIR}/models/beatthis.pt
    )
    set_tests_properties(beatit_torchscript_model_device_tests PROPERTIES
        SKIP_RETURN_CODE 77
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

install(TARGETS beatit beatit_lib)
if(BEATIT_USE_TORCH)
    install(TARGETS beatit_backend_torch_plugin
        LIBRARY DESTINATION lib/beatit
    )
endif()
