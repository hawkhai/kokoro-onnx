# Flutter Kokoro plugin - Windows build configuration.
# Bundles prebuilt espeak-ng DLL for phonemization.
cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME "flutter_kokoro")
project(${PROJECT_NAME} LANGUAGES CXX)

cmake_policy(VERSION 3.14...3.25)

set(PLUGIN_NAME "flutter_kokoro_plugin")

# --- Plugin library (method channel stub, required by Flutter tooling) ---

list(APPEND PLUGIN_SOURCES
  "flutter_kokoro_plugin.cpp"
  "flutter_kokoro_plugin.h"
)

add_library(${PLUGIN_NAME} SHARED
  "include/flutter_kokoro/flutter_kokoro_plugin_c_api.h"
  "flutter_kokoro_plugin_c_api.cpp"
  ${PLUGIN_SOURCES}
)

apply_standard_settings(${PLUGIN_NAME})

set_target_properties(${PLUGIN_NAME} PROPERTIES
  CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)

target_include_directories(${PLUGIN_NAME} INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(${PLUGIN_NAME} PRIVATE flutter flutter_wrapper_plugin)

# --- Bundled libraries ---
# espeak-ng DLL for phonemization (loaded via dart:ffi at runtime).
set(ESPEAK_NG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/espeak-ng")

set(flutter_kokoro_bundled_libraries
  "${ESPEAK_NG_DIR}/bin/espeak-ng.dll"
  PARENT_SCOPE
)

# --- Tests ---
if (${include_${PROJECT_NAME}_tests})
set(TEST_RUNNER "${PROJECT_NAME}_test")
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.11.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Disable installation of googletest" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(${TEST_RUNNER}
  test/flutter_kokoro_plugin_test.cpp
  ${PLUGIN_SOURCES}
)
apply_standard_settings(${TEST_RUNNER})
target_include_directories(${TEST_RUNNER} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${TEST_RUNNER} PRIVATE flutter_wrapper_plugin)
target_link_libraries(${TEST_RUNNER} PRIVATE gtest_main gmock)
add_custom_command(TARGET ${TEST_RUNNER} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${FLUTTER_LIBRARY}" $<TARGET_FILE_DIR:${TEST_RUNNER}>
)

include(GoogleTest)
gtest_discover_tests(${TEST_RUNNER})
endif()
