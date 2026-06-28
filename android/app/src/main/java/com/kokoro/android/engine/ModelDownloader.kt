package com.kokoro.android.engine

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

/**
 * 模型文件下载管理器
 *
 * 从 GitHub releases 下载 ONNX 模型和声音文件
 */
class ModelDownloader(private val context: Context) {

    companion object {
        private const val TAG = "ModelDownloader"

        // Model download URLs
        const val MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
        const val VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

        const val MODEL_FILENAME = "kokoro-v1.0.onnx"
        const val VOICES_FILENAME = "voices-v1.0.bin"

        // Expected file sizes (approximate)
        const val MODEL_SIZE = 80_000_000L  // ~80MB
        const val VOICES_SIZE = 300_000_000L  // ~300MB
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(300, TimeUnit.SECONDS)
        .followRedirects(true)
        .followSslRedirects(true)
        .build()

    /**
     * 下载进度回调
     */
    interface DownloadProgressCallback {
        fun onProgress(fileName: String, bytesRead: Long, totalBytes: Long)
        fun onComplete(fileName: String, file: File)
        fun onError(fileName: String, error: String)
    }

    /**
     * 检查模型文件是否已下载
     */
    fun areModelsDownloaded(): Boolean {
        val modelFile = getModelFile()
        val voicesFile = getVoicesFile()
        return modelFile.exists() && voicesFile.exists() &&
               modelFile.length() > 0 && voicesFile.length() > 0
    }

    /**
     * 获取模型文件路径
     */
    fun getModelFile(): File = File(context.filesDir, MODEL_FILENAME)

    /**
     * 获取声音文件路径
     */
    fun getVoicesFile(): File = File(context.filesDir, VOICES_FILENAME)

    /**
     * 下载所有模型文件
     */
    suspend fun downloadModels(
        callback: DownloadProgressCallback
    ) = withContext(Dispatchers.IO) {
        val modelFile = getModelFile()
        val voicesFile = getVoicesFile()

        // Download model file if needed
        if (!modelFile.exists() || modelFile.length() == 0L) {
            downloadFile(MODEL_URL, modelFile, callback)
        }

        // Download voices file if needed
        if (!voicesFile.exists() || voicesFile.length() == 0L) {
            downloadFile(VOICES_URL, voicesFile, callback)
        }
    }

    /**
     * 下载单个文件
     */
    private fun downloadFile(
        url: String,
        targetFile: File,
        callback: DownloadProgressCallback
    ) {
        val fileName = targetFile.name
        Log.d(TAG, "Downloading $fileName from $url")

        try {
            val request = Request.Builder()
                .url(url)
                .build()

            val response = client.newCall(request).execute()

            if (!response.isSuccessful) {
                throw IOException("Download failed: ${response.code} ${response.message}")
            }

            val body = response.body ?: throw IOException("Empty response body")
            val contentLength = body.contentLength()

            // Write to temp file first
            val tempFile = File(targetFile.parent, "$fileName.tmp")

            body.byteStream().use { input ->
                FileOutputStream(tempFile).use { output ->
                    val buffer = ByteArray(8192)
                    var bytesRead: Long = 0
                    var read: Int

                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                        bytesRead += read
                        callback.onProgress(fileName, bytesRead, contentLength)
                    }

                    output.flush()
                }
            }

            // Rename temp file to final file
            if (targetFile.exists()) {
                targetFile.delete()
            }
            tempFile.renameTo(targetFile)

            Log.d(TAG, "Downloaded $fileName: ${targetFile.length()} bytes")
            callback.onComplete(fileName, targetFile)

        } catch (e: Exception) {
            Log.e(TAG, "Download failed for $fileName", e)
            callback.onError(fileName, e.message ?: "Unknown error")
            throw e
        }
    }

    /**
     * 删除所有模型文件
     */
    fun deleteModels() {
        getModelFile().delete()
        getVoicesFile().delete()
        Log.d(TAG, "Deleted all model files")
    }

    /**
     * 获取文件大小描述
     */
    fun getFileSizeDescription(): String {
        val modelSize = if (getModelFile().exists()) getModelFile().length() else 0
        val voicesSize = if (getVoicesFile().exists()) getVoicesFile().length() else 0
        return "Model: ${formatSize(modelSize)}, Voices: ${formatSize(voicesSize)}"
    }

    private fun formatSize(bytes: Long): String {
        return when {
            bytes >= 1_000_000 -> "%.1f MB".format(bytes / 1_000_000.0)
            bytes >= 1_000 -> "%.1f KB".format(bytes / 1_000.0)
            else -> "$bytes B"
        }
    }
}
