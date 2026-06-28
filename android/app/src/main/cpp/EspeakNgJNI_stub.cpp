#include <jni.h>
#include <string>
#include <android/log.h>

#define TAG "EspeakNgJNI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)

/**
 * Stub JNI implementation when espeak-ng source is not available.
 * The app will work but phonemization will be disabled.
 * Users should provide pre-phonemized text.
 */

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_kokoro_android_engine_EspeakNgEngine_nativeInit(
        JNIEnv *env,
        jobject thiz,
        jstring dataPath) {
    LOGW("espeak-ng not compiled - phonemization disabled");
    LOGW("To enable: download espeak-ng source to app/src/main/cpp/espeak-ng/");
    return JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_com_kokoro_android_engine_EspeakNgEngine_nativePhonemize(
        JNIEnv *env,
        jobject thiz,
        jstring text,
        jstring lang) {
    LOGW("espeak-ng not compiled - returning original text");
    return text;
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
