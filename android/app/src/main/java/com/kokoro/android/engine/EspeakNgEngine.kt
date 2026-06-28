package com.kokoro.android.engine

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * espeak-ng 音素化引擎
 *
 * 通过 JNI 调用 espeak-ng 将文本转换为 IPA 音素
 */
class EspeakNgEngine(private val context: Context) {

    companion object {
        private const val TAG = "EspeakNgEngine"
        private const val ESPEAK_DATA_DIR = "espeak-ng-data"

        init {
            try {
                System.loadLibrary("kokoro_jni")
                Log.d(TAG, "Native library loaded")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library", e)
            }
        }
    }

    private var initialized = false

    /**
     * 初始化 espeak-ng 引擎
     * @return true if initialization succeeded
     */
    fun init(): Boolean {
        if (initialized) return true

        try {
            val dataPath = prepareDataPath()
            initialized = nativeInit(dataPath)
            Log.d(TAG, "Initialization result: $initialized")
            return initialized
        } catch (e: Exception) {
            Log.e(TAG, "Init failed", e)
            return false
        }
    }

    /**
     * 将文本转换为 IPA 音素
     * @param text 输入文本
     * @param lang 语言代码 (e.g., "en-us", "zh", "ja")
     * @return IPA 音素字符串
     */
    fun phonemize(text: String, lang: String = "en-us"): String {
        if (!initialized) {
            Log.e(TAG, "Engine not initialized")
            return ""
        }
        return nativePhonemize(text, lang)
    }

    /**
     * 检查引擎是否已初始化
     */
    fun isInitialized(): Boolean = initialized

    /**
     * 释放资源
     */
    fun destroy() {
        if (initialized) {
            nativeDestroy()
            initialized = false
        }
    }

    /**
     * 准备 espeak-ng 数据路径
     */
    private fun prepareDataPath(): String {
        val dataDir = File(context.filesDir, ESPEAK_DATA_DIR)
        if (!dataDir.exists()) {
            dataDir.mkdirs()
            copyAssetToDir(ESPEAK_DATA_DIR, dataDir)
        }
        return dataDir.absolutePath
    }

    /**
     * 从 assets 复制数据目录
     */
    private fun copyAssetToDir(assetDir: String, targetDir: File) {
        try {
            val assets = context.assets.list(assetDir) ?: return

            if (assets.isEmpty()) {
                // It's a file, copy it
                context.assets.open(assetDir).use { input ->
                    FileOutputStream(File(targetDir, assetDir)).use { output ->
                        input.copyTo(output)
                    }
                }
            } else {
                // It's a directory, recurse
                for (asset in assets) {
                    val subDir = File(targetDir, asset)
                    subDir.mkdirs()
                    copyAssetToDir("$assetDir/$asset", targetDir)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to copy asset: $assetDir", e)
        }
    }

    // Native methods
    private external fun nativeInit(dataPath: String): Boolean
    private external fun nativePhonemize(text: String, lang: String): String
    private external fun nativeDestroy()
    private external fun nativeIsInitialized(): Boolean
}
