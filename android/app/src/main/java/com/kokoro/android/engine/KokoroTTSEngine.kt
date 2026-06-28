package com.kokoro.android.engine

import android.content.Context
import android.util.Log
import com.kokoro.android.AudioTrimmer
import com.kokoro.android.KokoroConfig
import com.kokoro.android.NumpyLoader
import com.kokoro.android.Tokenizer
import java.nio.FloatBuffer

/**
 * 完整的 Kokoro TTS 引擎
 *
 * 整合了：
 * - ONNX 模型推理
 * - espeak-ng 音素化
 * - 声音管理
 * - 音频处理
 */
class KokoroTTSEngine(private val context: Context) {

    companion object {
        private const val TAG = "KokoroTTSEngine"
    }

    private var espeakEngine: EspeakNgEngine? = null
    private var onnxSession: ai.onnxruntime.OrtSession? = null
    private var ortEnv: ai.onnxruntime.OrtEnvironment? = null
    private var voices: Map<String, Array<FloatArray>> = emptyMap()
    private var tokenizer: Tokenizer = Tokenizer()
    private var initialized = false

    // Voice data loaded from numpy file
    private var voicesLoaded = false

    /**
     * 初始化引擎
     * @param modelPath ONNX 模型文件路径
     * @param voicesPath 声音文件路径 (numpy format)
     * @return true if initialization succeeded
     */
    fun init(modelPath: String, voicesPath: String): Boolean {
        try {
            Log.d(TAG, "Initializing TTS engine")
            Log.d(TAG, "Model: $modelPath")
            Log.d(TAG, "Voices: $voicesPath")

            // Initialize ONNX Runtime
            ortEnv = ai.onnxruntime.OrtEnvironment.getEnvironment()
            onnxSession = ortEnv!!.createSession(modelPath)
            Log.d(TAG, "ONNX model loaded")

            // Load voices
            voices = NumpyLoader.loadVoices(voicesPath)
            voicesLoaded = true
            Log.d(TAG, "Loaded ${voices.size} voices: ${voices.keys.sorted()}")

            // Initialize espeak-ng (optional, for phonemization)
            espeakEngine = EspeakNgEngine(context)
            val espeakReady = espeakEngine!!.init()
            if (espeakReady) {
                Log.d(TAG, "espeak-ng initialized for phonemization")
            } else {
                Log.w(TAG, "espeak-ng not available - use pre-phonemized text")
            }

            initialized = true
            return true

        } catch (e: Exception) {
            Log.e(TAG, "Initialization failed", e)
            release()
            return false
        }
    }

    /**
     * 检查引擎是否已初始化
     */
    fun isInitialized(): Boolean = initialized

    /**
     * 获取可用声音列表
     */
    fun getVoices(): List<String> = voices.keys.sorted()

    /**
     * 检查 espeak-ng 是否可用
     */
    fun isPhonemizationAvailable(): Boolean = espeakEngine?.isInitialized() == true

    /**
     * 将文本转换为音素
     * @param text 输入文本
     * @param lang 语言代码
     * @return IPA 音素字符串，如果 espeak-ng 不可用则返回原文
     */
    fun phonemize(text: String, lang: String = "en-us"): String {
        if (espeakEngine?.isInitialized() == true) {
            return espeakEngine!!.phonemize(text, lang)
        }
        Log.w(TAG, "Phonemization not available, returning original text")
        return text
    }

    /**
     * 生成语音
     *
     * @param text 输入文本或音素
     * @param voice 声音名称
     * @param speed 语速 (0.5 - 2.0)
     * @param lang 语言代码 (用于音素化)
     * @param isPhonemes true 表示输入已是音素，false 表示需要音素化
     * @param trimSilence 是否裁剪静音
     * @return 音频采样数据 (FloatArray) 和采样率
     */
    fun synthesize(
        text: String,
        voice: String = "af_sarah",
        speed: Float = 1.0f,
        lang: String = "en-us",
        isPhonemes: Boolean = false,
        trimSilence: Boolean = true
    ): Pair<FloatArray, Int> {
        require(initialized) { "Engine not initialized" }
        require(speed in 0.5f..2.0f) { "Speed must be between 0.5 and 2.0" }

        val voiceStyle = voices[voice]
            ?: throw IllegalArgumentException("Voice '$voice' not found")

        // Get phonemes
        val phonemes = if (isPhonemes) {
            text
        } else {
            phonemize(text, lang)
        }

        Log.d(TAG, "Synthesizing: $phonemes (voice=$voice, speed=$speed)")

        // Split into batches
        val batches = splitPhonemes(phonemes)
        Log.d(TAG, "Processing ${batches.size} batches")

        val audioParts = mutableListOf<FloatArray>()
        for (batch in batches) {
            val audio = runInference(batch, voiceStyle, speed)
            val trimmed = if (trimSilence) AudioTrimmer.trim(audio) else audio
            audioParts.add(trimmed)
        }

        val result = concatAudio(audioParts)
        val duration = result.size.toFloat() / KokoroConfig.SAMPLE_RATE
        Log.d(TAG, "Generated %.2fs audio".format(duration))

        return Pair(result, KokoroConfig.SAMPLE_RATE)
    }

    /**
     * 运行 ONNX 推理
     */
    private fun runInference(
        phonemes: String,
        voiceStyle: Array<FloatArray>,
        speed: Float
    ): FloatArray {
        val session = onnxSession ?: throw IllegalStateException("ONNX session not initialized")
        val env = ortEnv ?: throw IllegalStateException("ORT environment not initialized")

        val tokens = tokenizer.tokenize(phonemes)
        val tokenArray = longArrayOf(0L) + tokens.map { it.toLong() }.toLongArray() + longArrayOf(0L)

        val styleIndex = tokens.size
        if (styleIndex >= voiceStyle.size) {
            throw IllegalArgumentException(
                "Token length ($styleIndex) exceeds voice dimensions (${voiceStyle.size})"
            )
        }
        val style = voiceStyle[styleIndex]

        // Prepare tensors
        val inputIdsBuffer = java.nio.LongBuffer.allocate(tokenArray.size)
        inputIdsBuffer.put(tokenArray)
        inputIdsBuffer.rewind()

        val styleBuffer = FloatBuffer.allocate(style.size)
        styleBuffer.put(style)
        styleBuffer.rewind()

        val speedBuffer = FloatBuffer.allocate(1)
        speedBuffer.put(speed)
        speedBuffer.rewind()

        val inputIdsTensor = ai.onnxruntime.OnnxTensor.createTensor(
            env, inputIdsBuffer, longArrayOf(1, tokenArray.size.toLong())
        )
        val styleTensor = ai.onnxruntime.OnnxTensor.createTensor(
            env, styleBuffer, longArrayOf(style.size.toLong())
        )
        val speedTensor = ai.onnxruntime.OnnxTensor.createTensor(
            env, speedBuffer, longArrayOf(1)
        )

        try {
            val inputs = mapOf(
                "input_ids" to inputIdsTensor,
                "style" to styleTensor,
                "speed" to speedTensor
            )

            val results = session.run(inputs)
            val output = results[0].value

            return when (output) {
                is Array<*> -> {
                    @Suppress("UNCHECKED_CAST")
                    val nestedArray = output as Array<Array<Float>>
                    nestedArray[0].toFloatArray()
                }
                is FloatArray -> output
                else -> throw IllegalArgumentException("Unexpected output format: ${output?.javaClass}")
            }
        } finally {
            inputIdsTensor.close()
            styleTensor.close()
            speedTensor.close()
        }
    }

    /**
     * 分割音素为批次
     */
    private fun splitPhonemes(phonemes: String): List<String> {
        val words = phonemes.split(Regex("([.,!?;])")).filter { it.isNotBlank() }
        val batches = mutableListOf<String>()
        val currentBatch = StringBuilder()

        for (part in words) {
            val trimmed = part.trim()
            if (trimmed.isEmpty()) continue

            if (currentBatch.length + trimmed.length + 1 >= KokoroConfig.MAX_PHONEME_LENGTH) {
                if (currentBatch.isNotEmpty()) {
                    batches.add(currentBatch.toString().trim())
                    currentBatch.clear()
                }
                currentBatch.append(trimmed)
            } else {
                if (trimmed in ".,!?;") {
                    currentBatch.append(trimmed)
                } else {
                    if (currentBatch.isNotEmpty()) currentBatch.append(" ")
                    currentBatch.append(trimmed)
                }
            }
        }

        if (currentBatch.isNotEmpty()) {
            batches.add(currentBatch.toString().trim())
        }

        return batches
    }

    /**
     * 拼接音频数组
     */
    private fun concatAudio(arrays: List<FloatArray>): FloatArray {
        val totalSize = arrays.sumOf { it.size }
        val result = FloatArray(totalSize)
        var offset = 0
        for (array in arrays) {
            array.copyInto(result, offset)
            offset += array.size
        }
        return result
    }

    /**
     * 释放资源
     */
    fun release() {
        onnxSession?.close()
        ortEnv?.close()
        espeakEngine?.destroy()
        onnxSession = null
        ortEnv = null
        espeakEngine = null
        initialized = false
        voicesLoaded = false
        voices = emptyMap()
        Log.d(TAG, "Engine released")
    }
}
