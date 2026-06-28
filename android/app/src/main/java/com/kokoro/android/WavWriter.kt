package com.kokoro.android

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Utility for writing WAV audio files.
 */
object WavWriter {

    /**
     * Write audio samples to a WAV file.
     *
     * @param filePath Output file path
     * @param samples Audio samples as FloatArray (range -1.0 to 1.0)
     * @param sampleRate Sample rate in Hz (default: 24000)
     */
    fun write(filePath: String, samples: FloatArray, sampleRate: Int = KokoroConfig.SAMPLE_RATE) {
        val file = File(filePath)
        val numChannels = 1
        val bitsPerSample = 16
        val byteRate = sampleRate * numChannels * bitsPerSample / 8
        val blockAlign = numChannels * bitsPerSample / 8
        val dataSize = samples.size * bitsPerSample / 8

        FileOutputStream(file).use { fos ->
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)

            // RIFF header
            header.put("RIFF".toByteArray())
            header.putInt(36 + dataSize) // file size - 8
            header.put("WAVE".toByteArray())

            // fmt chunk
            header.put("fmt ".toByteArray())
            header.putInt(16) // chunk size
            header.putShort(1) // PCM format
            header.putShort(numChannels.toShort())
            header.putInt(sampleRate)
            header.putInt(byteRate)
            header.putShort(blockAlign.toShort())
            header.putShort(bitsPerSample.toShort())

            // data chunk
            header.put("data".toByteArray())
            header.putInt(dataSize)

            fos.write(header.array())

            // Write audio data as 16-bit PCM
            val dataBuffer = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN)
            for (sample in samples) {
                // Clamp to [-1, 1] and convert to 16-bit integer
                val clamped = sample.coerceIn(-1f, 1f)
                val intSample = (clamped * 32767f).toInt().toShort()
                dataBuffer.putShort(intSample)
            }

            fos.write(dataBuffer.array())
        }
    }

    /**
     * Convert audio samples to 16-bit PCM byte array for playback.
     */
    fun toPcm16(samples: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (sample in samples) {
            val clamped = sample.coerceIn(-1f, 1f)
            val intSample = (clamped * 32767f).toInt().toShort()
            buffer.putShort(intSample)
        }
        return buffer.array()
    }
}
