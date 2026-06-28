# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in /sdk/tools/proguard/proguard-android.txt

# Keep ONNX Runtime classes
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# Keep Kokoro TTS classes
-keep class com.kokoro.android.** { *; }
