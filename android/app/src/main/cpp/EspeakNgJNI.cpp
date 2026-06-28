#include <jni.h>
#include <string>
#include <cstring>
#include <android/log.h>

#define TAG "EspeakNgJNI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#ifdef USE_ESPEAK_NG
#include "espeak-ng/espeak_ng.h"
#include "espeak-ng/speak_lib.h"

static bool initialized = false;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_kokoro_android_engine_EspeakNgEngine_nativeInit(
        JNIEnv *env,
        jobject thiz,
        jstring dataPath) {
    const char *path = env->GetStringUTFChars(dataPath, nullptr);
    LOGD("Initializing espeak-ng with data path: %s", path);

    espeak_AUDIO_OUTPUT output = AUDIO_OUTPUT_SYNCHRONOUS;
    int buflength = 500;
    const char *options = "";
    int result;

    result = espeak_Initialize(output, buflength, path, 0);
    if (result < 0) {
        LOGE("Failed to initialize espeak-ng: %d", result);
        env->ReleaseStringUTFChars(dataPath, path);
        return JNI_FALSE;
    }

    initialized = true;
    LOGD("espeak-ng initialized successfully, sample rate: %d", result);
    env->ReleaseStringUTFChars(dataPath, path);
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_com_kokoro_android_engine_EspeakNgEngine_nativePhonemize(
        JNIEnv *env,
        jobject thiz,
        jstring text,
        jstring lang) {
    if (!initialized) {
        LOGE("espeak-ng not initialized");
        return env->NewStringUTF("");
    }

    const char *textStr = env->GetStringUTFChars(text, nullptr);
    const char *langStr = env->GetStringUTFChars(lang, nullptr);

    LOGD("Phonemizing text: %s, lang: %s", textStr, langStr);

    // Set language
    espeak_VOICE voice;
    memset(&voice, 0, sizeof(voice));
    voice.languages = langStr;
    espeak_SetVoiceByProperties(&voice);

    // Phonemize text
    // espeak-ng phonemize function
    char *phonemes = nullptr;
    int *unique_ids = nullptr;
    int text_len = strlen(textStr);
    unsigned int flags = espeakPHONEMES_IPA | espeakSTRESS | espeakCHARS_UTF8;

    espeak_ng_TEXT_TO_PHONEMES(
        (const void *)textStr,
        text_len + 1,
        flags,
        &phonemes,
        &unique_ids
    );

    jstring result;
    if (phonemes != nullptr) {
        LOGD("Phonemes: %s", phonemes);
        result = env->NewStringUTF(phonemes);
        free(phonemes);
    } else {
        LOGE("Phonemization failed");
        result = env->NewStringUTF("");
    }

    if (unique_ids != nullptr) {
        free(unique_ids);
    }

    env->ReleaseStringUTFChars(text, textStr);
    env->ReleaseStringUTFChars(lang, langStr);

    return result;
}

JNIEXPORT void JNICALL
Java_com_kokoro_android_engine_EspeakNgEngine_nativeDestroy(
        JNIEnv *env,
        jobject thiz) {
    if (initialized) {
        espeak_Terminate();
        initialized = false;
        LOGD("espeak-ng terminated");
    }
}

JNIEXPORT jboolean JNICALL
Java_com_kokoro_android_engine_EspeakNgEngine_nativeIsInitialized(
        JNIEnv *env,
        jobject thiz) {
    return initialized ? JNI_TRUE : JNI_FALSE;
}

} // extern "C"

#else // USE_ESPEAK_NG not defined

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_kokoro_android_engine_EspeakNgEngine_nativeInit(
        JNIEnv *env,
        jobject thiz,
        jstring dataPath) {
    LOGE("espeak-ng not compiled - nativeInit not available");
    return JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_com_kokoro_android_engine_EspeakNgEngine_nativePhonemize(
        JNIEnv *env,
        jobject thiz,
        jstring text,
        jstring lang) {
    LOGE("espeak-ng not compiled - nativePhonemize not available");
    return env->NewStringUTF("");
}

JNIEXPORT void JNICALL
Java_com_kokoro_android_engine_EspeakNgEngine_nativeDestroy(
        JNIEnv *env,
        jobject thiz) {
    // No-op
}

JNIEXPORT jboolean JNICALL
Java_com_kokoro_android_engine_EspeakNgEngine_nativeIsInitialized(
        JNIEnv *env,
        jobject thiz) {
    return JNI_FALSE;
}

} // extern "C"

#endif // USE_ESPEAK_NG
