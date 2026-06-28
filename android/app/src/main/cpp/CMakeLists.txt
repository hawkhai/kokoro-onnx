cmake_minimum_required(VERSION 3.22.1)
project("kokoro_android")

# espeak-ng source directory - download from https://github.com/espeak-ng/espeak-ng
set(ESPEAK_NG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/espeak-ng")

# If espeak-ng source exists, build it
if(EXISTS "${ESPEAK_NG_DIR}/CMakeLists.txt")
    message(STATUS "Building espeak-ng from source")

    # Include espeak-ng headers
    include_directories(
        ${ESPEAK_NG_DIR}/src/include
        ${ESPEAK_NG_DIR}/src/ucd
    )

    # espeak-ng source files
    set(ESPEAK_SOURCES
        ${ESPEAK_NG_DIR}/src/libespeak-ng/common.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/dictionary.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/encoding.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/error.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/espeak_api.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/espeak_ng.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/event.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/fifo.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/ieee80.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/intonation.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/klatt.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/laugh.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/mbrola.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/mnemonics.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/numbers.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/phoneme.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/phonemelist.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/readclause.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/resolve.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/setlengths.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/soundicon.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/spect.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/ssml.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/synthdata.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/synthesize.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/synth_mbrola.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/tr_languages.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/translate.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/translateword.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/voices.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/wavegen.c
        ${ESPEAK_NG_DIR}/src/libespeak-ng/synth_so.c
    )

    # Add UCD source
    set(UCD_SOURCES
        ${ESPEAK_NG_DIR}/src/ucd/ucd.c
    )

    # Create espeak-ng static library
    add_library(espeak-ng STATIC ${ESPEAK_SOURCES} ${UCD_SOURCES})
    target_compile_definitions(espeak-ng PRIVATE
        LIBESPEAK_NG_EXPORT=1
        PATH_ESPEAK_DATA="/data/data/com.kokoro.android/files/espeak-ng-data"
        USE_KLATT=1
        USE_LIBPCAUDIO=0
        INCLUDE_KLATT=1
        INCLUDE_SPEECHPLAYER=0
        HYMN_ESPEAK_VERSION="1.0"
        PACKAGE_VERSION="1.52-dev"
    )
    target_include_directories(espeak-ng PUBLIC
        ${ESPEAK_NG_DIR}/src/include
        ${ESPEAK_NG_DIR}/src/ucd
    )

    # JNI wrapper
    add_library(kokoro_jni SHARED EspeakNgJNI.cpp)
    target_link_libraries(kokoro_jni espeak-ng log)

else()
    message(STATUS "espeak-ng source not found, building stub JNI")
    message(STATUS "To enable phonemization, download espeak-ng source to: ${ESPEAK_NG_DIR}")

    # Build stub JNI (no phonemization)
    add_library(kokoro_jni SHARED EspeakNgJNI_stub.cpp)
    target_link_libraries(kokoro_jni log)
endif()
