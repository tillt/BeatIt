cmake_minimum_required(VERSION 3.20)

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

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BEATIT_USE_TORCH "Enable Torch backend" ON)
set(BEATIT_TORCH_ROOT "" CACHE PATH "Path to LibTorch root")

if(BEATIT_USE_TORCH AND BEATIT_TORCH_ROOT STREQUAL "")
    if(EXISTS "/opt/homebrew/opt/libtorch/share/cmake/Torch")
        set(BEATIT_TORCH_ROOT "/opt/homebrew/opt/libtorch" CACHE PATH "Path to LibTorch root" FORCE)
    elseif(EXISTS "/usr/local/opt/libtorch/share/cmake/Torch")
        set(BEATIT_TORCH_ROOT "/usr/local/opt/libtorch" CACHE PATH "Path to LibTorch root" FORCE)
    endif()
endif()

set(BEATIT_LIB_SOURCES
    src/beatit/analysis.cpp
    src/beatit/analysis_bpm_beats.cpp
    src/beatit/analysis_bpm_activation.cpp
    src/beatit/analysis_backend.cpp
    src/beatit/audio_dsp.cpp
    src/beatit/audio_features.cpp
    src/beatit/logging.cpp
    src/beatit/version.cpp
    src/beatit/coreml_inference.mm
    src/beatit/coreml_model_utils.mm
    src/beatit/coreml_metadata.mm
    src/beatit/pp_pipeline.cpp
    src/beatit/pp_helpers.cpp
    src/beatit/pp_dbn.cpp
    src/beatit/pp_dbn_decode.cpp
    src/beatit/pp_dbn_anchor.cpp
    src/beatit/pp_dbn_grid.cpp
    src/beatit/pp_dbn_tempo.cpp
    src/beatit/pp_logits.cpp
    src/beatit/pp_grid_fill.cpp
    src/beatit/pp_result_ops.cpp
    src/beatit/pp_tempo_fit.cpp
    src/beatit/pp_window.cpp
    src/beatit/coreml_preset.cpp
    src/beatit/dbn_calmdad.cpp
    src/beatit/dbn_calmdad_sparse.cpp
    src/beatit/dbn_beatit.cpp
    src/beatit/refiner.cpp
    src/beatit/refiner_constant.cpp
    src/beatit/refiner_constant_regions.cpp
    src/beatit/activation_merge.cpp
    src/beatit/stream_finalize.cpp
    src/beatit/inference_backend_coreml.cpp
    src/beatit/sparse_probe.cpp
    src/beatit/sparse_probe_pick.cpp
    src/beatit/sparse_probe_score.cpp
    src/beatit/sparse_phase_metrics.cpp
    src/beatit/sparse_refinement.cpp
    src/beatit/sparse_edge.cpp
    src/beatit/sparse_edge_adjust.cpp
    src/beatit/sparse_edge_metrics.cpp
    src/beatit/sparse_edge_phase.cpp
    src/beatit/sparse_waveform.cpp
    src/beatit/stream.cpp
)
if(BEATIT_USE_TORCH)
    list(APPEND BEATIT_LIB_SOURCES
        src/beatit/analysis_torch.cpp
        src/beatit/inference_backend_torch.cpp
        src/beatit/torch_mel.cpp
    )
else()
    list(APPEND BEATIT_LIB_SOURCES
        src/beatit/analysis_torch_stub.cpp
        src/beatit/inference_backend_torch_stub.cpp
    )
endif()

add_library(beatit_lib ${BEATIT_LIB_SOURCES})

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
    set(Torch_DIR "${BEATIT_TORCH_ROOT}/torch/share/cmake/Torch")
    find_package(Torch REQUIRED)
    target_include_directories(${target} PRIVATE ${TORCH_INCLUDE_DIRS})
    set(BEATIT_TORCH_LIBS
        torch
        torch_cpu
        c10
    )
    find_library(TORCH_GLOBAL_DEPS_LIB
        NAMES torch_global_deps
        PATHS "${BEATIT_TORCH_ROOT}/torch/lib"
        NO_DEFAULT_PATH
    )
    if(TORCH_GLOBAL_DEPS_LIB)
        list(APPEND BEATIT_TORCH_LIBS "${TORCH_GLOBAL_DEPS_LIB}")
    endif()
    target_link_libraries(${target} PRIVATE ${BEATIT_TORCH_LIBS})
    target_compile_definitions(${target} PRIVATE BEATIT_USE_TORCH=1)
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH "${BEATIT_TORCH_ROOT}/torch/lib"
    )
endfunction()

beatit_link_torch(beatit_lib)

add_executable(beatit
    src/cli/main.mm
)

target_link_libraries(beatit
    PRIVATE
        beatit_lib
)

if(APPLE)
    target_compile_options(beatit_lib PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(beatit_lib PRIVATE -fobjc-arc)
    target_compile_options(beatit PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_lib PRIVATE "-framework Accelerate" "-framework CoreML" "-framework Foundation")
    target_link_libraries(beatit PRIVATE "-framework AVFoundation" "-framework Foundation" "-framework CoreMedia" "-framework AudioToolbox")
endif()

if(APPLE)
    file(GLOB BEATIT_PUBLIC_HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/beatit/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/beatit/*.hpp")

    set(BEATIT_FRAMEWORK_SOURCES
        src/beatit/analysis.cpp
        src/beatit/analysis_bpm_beats.cpp
        src/beatit/analysis_bpm_activation.cpp
        src/beatit/analysis_backend.cpp
        src/beatit/audio_dsp.cpp
        src/beatit/audio_features.cpp
        src/beatit/logging.cpp
        src/beatit/version.cpp
        src/beatit/coreml_inference.mm
        src/beatit/coreml_model_utils.mm
        src/beatit/coreml_metadata.mm
        src/beatit/pp_pipeline.cpp
        src/beatit/pp_helpers.cpp
        src/beatit/pp_dbn.cpp
        src/beatit/pp_dbn_decode.cpp
        src/beatit/pp_dbn_anchor.cpp
        src/beatit/pp_dbn_grid.cpp
        src/beatit/pp_dbn_tempo.cpp
        src/beatit/pp_logits.cpp
        src/beatit/pp_grid_fill.cpp
        src/beatit/pp_result_ops.cpp
        src/beatit/pp_tempo_fit.cpp
        src/beatit/pp_window.cpp
        src/beatit/coreml_preset.cpp
        src/beatit/dbn_calmdad.cpp
        src/beatit/dbn_calmdad_sparse.cpp
        src/beatit/dbn_beatit.cpp
        src/beatit/refiner.cpp
        src/beatit/refiner_constant.cpp
        src/beatit/refiner_constant_regions.cpp
        src/beatit/activation_merge.cpp
        src/beatit/stream_finalize.cpp
        src/beatit/inference_backend_coreml.cpp
        src/beatit/sparse_probe.cpp
        src/beatit/sparse_probe_pick.cpp
        src/beatit/sparse_probe_score.cpp
        src/beatit/sparse_phase_metrics.cpp
        src/beatit/sparse_refinement.cpp
        src/beatit/sparse_edge.cpp
        src/beatit/sparse_edge_adjust.cpp
        src/beatit/sparse_edge_metrics.cpp
        src/beatit/sparse_edge_phase.cpp
        src/beatit/sparse_waveform.cpp
        src/beatit/stream.cpp
        ${BEATIT_PUBLIC_HEADERS}
    )
    if(BEATIT_USE_TORCH)
        list(APPEND BEATIT_FRAMEWORK_SOURCES
            src/beatit/analysis_torch.cpp
            src/beatit/inference_backend_torch.cpp
            src/beatit/torch_mel.cpp
        )
    else()
        list(APPEND BEATIT_FRAMEWORK_SOURCES
            src/beatit/analysis_torch_stub.cpp
            src/beatit/inference_backend_torch_stub.cpp
        )
    endif()

    add_library(beatit_framework SHARED ${BEATIT_FRAMEWORK_SOURCES})

    target_include_directories(beatit_framework
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/generated
    )

    beatit_link_torch(beatit_framework)

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

    add_custom_command(TARGET beatit_framework POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/models/beatit.mlmodelc"
            "$<TARGET_FILE_DIR:beatit_framework>/Resources/beatit.mlmodelc"
        COMMENT "Copying CoreML model into BeatIt.framework resources"
    )

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/coreml_out_latest/BeatThis_small0.mlpackage")
        add_custom_command(TARGET beatit_framework POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/coreml_out_latest/BeatThis_small0.mlpackage"
                "$<TARGET_FILE_DIR:beatit_framework>/Resources/BeatThis_small0.mlpackage"
            COMMENT "Copying BeatThis small CoreML package into BeatIt.framework resources"
        )
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/coreml_out_latest/BeatThis_small0.mlmodelc")
        add_custom_command(TARGET beatit_framework POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/coreml_out_latest/BeatThis_small0.mlmodelc"
                "$<TARGET_FILE_DIR:beatit_framework>/Resources/BeatThis_small0.mlmodelc"
            COMMENT "Copying BeatThis small CoreML model into BeatIt.framework resources"
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

    if(BEATIT_USE_TORCH AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/beatthis_mps.pt")
        add_custom_command(TARGET beatit_framework POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/models/beatthis_mps.pt"
                "$<TARGET_FILE_DIR:beatit_framework>/Resources/beatthis_mps.pt"
            COMMENT "Copying BeatThis MPS Torch model into BeatIt.framework resources"
        )
    endif()

    add_custom_command(TARGET beatit_framework POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:beatit_framework>/Headers/beatit"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/include/beatit"
            "$<TARGET_FILE_DIR:beatit_framework>/Headers/beatit"
        COMMAND ${CMAKE_COMMAND} -E rm -f
            "$<TARGET_FILE_DIR:beatit_framework>/Headers/analysis.h"
            "$<TARGET_FILE_DIR:beatit_framework>/Headers/coreml.h"
            "$<TARGET_FILE_DIR:beatit_framework>/Headers/coreml_preset.h"
            "$<TARGET_FILE_DIR:beatit_framework>/Headers/refiner.h"
            "$<TARGET_FILE_DIR:beatit_framework>/Headers/stream.h"
        COMMENT "Copying BeatIt headers into framework subfolder"
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

add_executable(beatit_dbn_purelove_tests
    tests/dbn_purelove_test.mm
)

target_link_libraries(beatit_dbn_purelove_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_dbn_purelove_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/beatit.mlpackage"
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
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/beatit.mlpackage"
    BEATIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
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
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/beatit.mlpackage"
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
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/beatit.mlpackage"
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
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/beatit.mlpackage"
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

add_executable(beatit_bazaar_window_alignment_sparse_tests
    tests/bazaar_window_alignment_test.mm
)

target_link_libraries(beatit_bazaar_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_bazaar_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/beatit.mlpackage"
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
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/beatit.mlpackage"
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

add_executable(beatit_neural_window_alignment_sparse_tests
    tests/neural_window_alignment_test.mm
)

target_link_libraries(beatit_neural_window_alignment_sparse_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_neural_window_alignment_sparse_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/beatit.mlpackage"
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

if(BEATIT_USE_TORCH)
    add_executable(beatit_beatthis_canonical_tests
        tests/beatthis_canonical_test.mm
    )

    target_link_libraries(beatit_beatthis_canonical_tests
        PRIVATE
            beatit_lib
    )

    beatit_link_torch(beatit_beatthis_canonical_tests)

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

install(TARGETS beatit beatit_lib)
