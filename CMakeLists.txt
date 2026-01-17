cmake_minimum_required(VERSION 3.20)

project(beatit
    VERSION 0.1.0
    DESCRIPTION "A Minimal BPM/beat tracker using CoreML. Lightweight, permissive."
    LANGUAGES CXX OBJCXX)

if(APPLE AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(beatit_lib
    src/beatit/analysis.cpp
    src/beatit/coreml.mm
    src/beatit/refiner.cpp
    src/beatit/stream.cpp
)

target_include_directories(beatit_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

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
    file(GLOB BEATIT_PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/beatit/*.h")

    add_library(beatit_framework SHARED
        src/beatit/analysis.cpp
        src/beatit/coreml.mm
        src/beatit/refiner.cpp
        src/beatit/stream.cpp
        ${BEATIT_PUBLIC_HEADERS}
    )

    target_include_directories(beatit_framework
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
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
            MACOSX_PACKAGE_LOCATION "Headers"
        )
    endforeach()

    add_custom_command(TARGET beatit_framework POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/models/beatit.mlmodelc"
            "$<TARGET_FILE_DIR:beatit_framework>/Resources/beatit.mlmodelc"
        COMMENT "Copying CoreML model into BeatIt.framework resources"
    )

    target_compile_options(beatit_framework PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_framework PRIVATE "-framework Accelerate" "-framework CoreML" "-framework Foundation")
endif()

enable_testing()

find_package(Python3 COMPONENTS Interpreter)

set(BEATIT_SYNTH_AUDIO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/training/generated")
set(BEATIT_SYNTH_AUDIO_120 "${BEATIT_SYNTH_AUDIO_DIR}/synthetic_click_120bpm.wav")
set(BEATIT_SYNTH_AUDIO_90 "${BEATIT_SYNTH_AUDIO_DIR}/synthetic_click_90bpm.wav")
set(BEATIT_SYNTH_AUDIO_TEST "${CMAKE_CURRENT_SOURCE_DIR}/training/test.wav")

if(Python3_Interpreter_FOUND)
    add_custom_command(
        OUTPUT
            "${BEATIT_SYNTH_AUDIO_TEST}"
            "${BEATIT_SYNTH_AUDIO_120}"
            "${BEATIT_SYNTH_AUDIO_90}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/training"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${BEATIT_SYNTH_AUDIO_DIR}"
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/training/generate_test_audio.py"
            --out "${BEATIT_SYNTH_AUDIO_TEST}"
            --bpm 123
            --duration 20
            --sample-rate 44100
            --frequency 200
            --click-ms 10
            --amplitude 0.5
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/training/generate_test_audio.py"
            --out "${BEATIT_SYNTH_AUDIO_120}" --bpm 120 --duration 10
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/training/generate_test_audio.py"
            --out "${BEATIT_SYNTH_AUDIO_90}" --bpm 90 --duration 10
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/training/generate_test_audio.py"
        COMMENT "Generating synthetic test audio"
    )

    add_custom_target(generate_test_audio
        DEPENDS
            "${BEATIT_SYNTH_AUDIO_TEST}"
            "${BEATIT_SYNTH_AUDIO_120}"
            "${BEATIT_SYNTH_AUDIO_90}"
    )
else()
    message(STATUS "Python3 not found; synthetic test audio will not be generated.")
endif()

add_executable(beatit_streaming_tests
    tests/streaming_ml_test.mm
)

target_link_libraries(beatit_streaming_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_streaming_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/beatit.mlpackage"
)

if(APPLE)
    target_compile_options(beatit_streaming_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_streaming_tests PRIVATE "-framework CoreML" "-framework Foundation")
endif()

if(TARGET generate_test_audio)
    add_dependencies(beatit_streaming_tests generate_test_audio)
endif()

add_test(NAME beatit_streaming_tests COMMAND beatit_streaming_tests)
set_tests_properties(beatit_streaming_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

add_executable(beatit_streaming_mp3_tests
    tests/streaming_ml_mp3_test.mm
)

target_link_libraries(beatit_streaming_mp3_tests
    PRIVATE
        beatit_lib
)

target_compile_definitions(beatit_streaming_mp3_tests PRIVATE
    BEATIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/beatit.mlpackage"
    BEATIT_TEST_AUDIO_PATH="${CMAKE_CURRENT_SOURCE_DIR}/training/test.wav"
)

if(APPLE)
    target_compile_options(beatit_streaming_mp3_tests PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(beatit_streaming_mp3_tests PRIVATE "-framework AVFoundation" "-framework CoreMedia" "-framework CoreML" "-framework Foundation")
endif()

if(TARGET generate_test_audio)
    add_dependencies(beatit_streaming_mp3_tests generate_test_audio)
endif()

add_test(NAME beatit_streaming_mp3_tests COMMAND beatit_streaming_mp3_tests)
set_tests_properties(beatit_streaming_mp3_tests PROPERTIES
    ENVIRONMENT
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/coreml_tmp;HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_home;XDG_CACHE_HOME=${CMAKE_CURRENT_BINARY_DIR}/coreml_cache"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SKIP_RETURN_CODE 77
)

install(TARGETS beatit beatit_lib)
