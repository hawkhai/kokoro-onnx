package com.kokoro.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Kokoro TTS Engine for Android.
 * Ported from Python kokoro_onnx/__init__.py
 *
 * Uses ONNX Runtime for model inference to generate speech from text/phonemes.
 *
 * Usage:
 * 1. Place model file (kokoro-v1.0.onnx) in assets or external storage
 * 2. Place voices file (voices-v1.0.bin as .npz) in assets or external storage
 * 3. Initialize: val kokoro = Kokoro(context, modelPath, voicesPath)
 * 4. Generate: val (audio, sampleRate) = kokoro.create("Hello", voice="af_sarah", lang="en-us")
 *
 * Note: Phonemization (text -> phonemes) requires espeak-ng.
 * For pre-phonemized text, use isPhonemes=true.
 */
class Kokoro(
    private val context: Context,
    modelPath: String,
    voicesPath: String,
    private val vocab: Map<String, Int> = Tokenizer.DEFAULT_VOCAB
) {
    companion object {
        private const val TAG = "KokoroTTS"
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val voices: Map<String, Array<FloatArray>>
    private val tokenizer: Tokenizer

    init {
        Log.d(TAG, "Initializing Kokoro TTS engine")
        Log.d(TAG, "Model path: $modelPath")
        Log.d(TAG, "Voices path: $voicesPath")

        // Load ONNX model
        session = env.createSession(modelPath)
        Log.d(TAG, "ONNX model loaded successfully")

        // Load voices
        voices = NumpyLoader.loadVoices(voicesPath)
        Log.d(TAG, "Loaded ${voices.size} voices: ${voices.keys.sorted()}")

        // Initialize tokenizer
        tokenizer = Tokenizer(vocab)
        Log.d(TAG, "Tokenizer initialized with ${vocab.size} vocabulary entries")
    }

    /**
     * Get list of available voice names.
     */
    fun getVoices(): List<String> = voices.keys.sorted()

    /**
     * Get voice style array for a given voice name.
     */
    fun getVoiceStyle(name: String): Array<FloatArray> {
        return voices[name] ?: throw IllegalArgumentException("Voice '$name' not found")
    }

    /**
     * Create audio from text.
     *
     * @param text Input text or phonemes
     * @param voice Voice name (e.g., "af_sarah") or pre-loaded voice style
     * @param speed Speech speed (0.5 to 2.0, default 1.0)
     * @param lang Language code (default "en-us")
     * @param isPhonemes If true, treat input as pre-phonemized text
     * @param trim Whether to trim silence from audio (default true)
     * @return Pair of (audio samples as FloatArray, sample rate)
     */
    fun create(
        text: String,
        voice: Any,
        speed: Float = 1.0f,
        lang: String = "en-us",
        isPhonemes: Boolean = false,
        trim: Boolean = true
    ): Pair<FloatArray, Int> {
        require(speed in 0.5f..2.0f) { "Speed must be between 0.5 and 2.0" }

        val voiceStyle = when (voice) {
            is String -> getVoiceStyle(voice)
            is Array<*> -> voice as Array<FloatArray>
            else -> throw IllegalArgumentException("Voice must be a String name or Array<FloatArray>")
        }

        val startTime = System.currentTimeMillis()

        // Get phonemes
        val phonemes = if (isPhonemes) {
            text
        } else {
            // For Android, we need to handle phonemization differently
            // The user should provide pre-phonemized text or use a phonemizer library
            Log.w(TAG, "Phonemization not available on Android. Use isPhonemes=true with pre-phonemized text.")
            text
        }

        // Split phonemes into batches
        val batchedPhonemes = splitPhonemes(phonemes)
        Log.d(TAG, "Processing ${batchedPhonemes.size} batches for ${phonemes.length} phonemes")

        val audioParts = mutableListOf<FloatArray>()
        for (batch in batchedPhonemes) {
            val audioPart = createAudio(batch, voiceStyle, speed)
            val trimmedPart = if (trim) {
                AudioTrimmer.trim(audioPart)
            } else {
                audioPart
            }
            audioParts.add(trimmedPart)
        }

        val audio = concatAudio(audioParts)
        val elapsed = System.currentTimeMillis() - startTime
        val audioDuration = audio.size.toFloat() / KokoroConfig.SAMPLE_RATE
        Log.d(TAG, "Created ${audioDuration}s audio in ${elapsed}ms")

        return Pair(audio, KokoroConfig.SAMPLE_RATE)
    }

    /**
     * Create audio from a single batch of phonemes.
     */
    private fun createAudio(
        phonemes: String,
        voiceStyle: Array<FloatArray>,
        speed: Float
    ): FloatArray {
        Log.d(TAG, "Phonemes: $phonemes")

        val tokens = tokenizer.tokenize(phonemes)
        val tokenArray = longArrayOf(0L) + tokens.map { it.toLong() }.toLongArray() + longArrayOf(0L)

        // Get voice style for this token length
        val styleIndex = tokens.size
        if (styleIndex >= voiceStyle.size) {
            throw IllegalArgumentException(
                "Token length ($styleIndex) exceeds voice style dimensions (${voiceStyle.size})"
            )
        }
        val style = voiceStyle[styleIndex]

        // Prepare inputs
        val inputIdsBuffer = LongBuffer.allocate(tokenArray.size)
        inputIdsBuffer.put(tokenArray)
        inputIdsBuffer.rewind()

        val styleBuffer = FloatBuffer.allocate(style.size)
        styleBuffer.put(style)
        styleBuffer.rewind()

        val speedBuffer = FloatBuffer.allocate(1)
        speedBuffer.put(speed)
        speedBuffer.rewind()

        // Create tensors
        val inputIdsTensor = OnnxTensor.createTensor(
            env,
            inputIdsBuffer,
            longArrayOf(1, tokenArray.size.toLong())
        )
        val styleTensor = OnnxTensor.createTensor(
            env,
            styleBuffer,
            longArrayOf(style.size.toLong())
        )
        val speedTensor = OnnxTensor.createTensor(
            env,
            speedBuffer,
            longArrayOf(1)
        )

        // Run inference
        val inputs = mapOf(
            "input_ids" to inputIdsTensor,
            "style" to styleTensor,
            "speed" to speedTensor
        )

        val results = session.run(inputs)
        val output = results[0].value

        // Clean up tensors
        inputIdsTensor.close()
        styleTensor.close()
        speedTensor.close()
        results.close()

        // Extract audio from output
        return when (output) {
            is Array<*> -> {
                @Suppress("UNCHECKED_CAST")
                val nestedArray = output as Array<Array<Float>>
                nestedArray[0].map { it }.toFloatArray()
            }
            is FloatArray -> output
            else -> {
                // Try to convert from various possible output formats
                val array = output as? Array<Any>
                if (array != null && array.isNotEmpty()) {
                    val inner = array[0]
                    if (inner is FloatArray) inner
                    else throw IllegalArgumentException("Unexpected output format: ${output::class.java}")
                } else {
                    throw IllegalArgumentException("Unexpected output format: ${output::class.java}")
                }
            }
        }
    }

    /**
     * Split phonemes into batches of MAX_PHONEME_LENGTH.
     * Prefers splitting at punctuation marks.
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
                    if (currentBatch.isNotEmpty()) {
                        currentBatch.append(" ")
                    }
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
     * Concatenate multiple audio arrays into one.
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
     * Clean up resources.
     */
    fun close() {
        session.close()
        env.close()
    }
}
