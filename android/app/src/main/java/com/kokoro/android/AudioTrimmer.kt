package com.kokoro.android

import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Audio trimming utility.
 * Ported from Python kokoro_onnx/trim.py (extracted from librosa).
 *
 * Trims leading and trailing silence from audio signals.
 */
object AudioTrimmer {

    /**
     * Trim leading and trailing silence from an audio signal.
     *
     * @param audio Float array of audio samples
     * @param topDb Threshold in dB below reference to consider as silence (default: 60)
     * @return Trimmed audio array
     */
    fun trim(audio: FloatArray, topDb: Float = 60f): FloatArray {
        val frameLength = 2048
        val hopLength = 512

        val nonSilent = signalToFrameNonsilent(audio, frameLength, hopLength, topDb)
        val nonzero = nonSilent.indices.filter { nonSilent[it] }

        if (nonzero.isEmpty()) {
            return FloatArray(0)
        }

        val start = framesToSamples(nonzero.first(), hopLength)
        val end = min(audio.size, framesToSamples(nonzero.last() + 1, hopLength))

        return audio.copyOfRange(start, end)
    }

    private fun signalToFrameNonsilent(
        y: FloatArray,
        frameLength: Int,
        hopLength: Int,
        topDb: Float
    ): BooleanArray {
        val rmsValues = rms(y, frameLength, hopLength)
        val maxRms = rmsValues.maxOrNull() ?: 1f

        return BooleanArray(rmsValues.size) { i ->
            val db = if (rmsValues[i] > 0f && maxRms > 0f) {
                20f * log10(rmsValues[i] / maxRms)
            } else {
                -100f
            }
            db > -topDb
        }
    }

    private fun rms(y: FloatArray, frameLength: Int, hopLength: Int): FloatArray {
        // Pad the signal
        val padSize = frameLength / 2
        val padded = FloatArray(y.size + 2 * padSize)
        y.copyInto(padded, padSize)

        val numFrames = (padded.size - frameLength) / hopLength + 1
        val result = FloatArray(numFrames)

        for (i in 0 until numFrames) {
            val start = i * hopLength
            var sumSq = 0f
            for (j in 0 until frameLength) {
                val sample = padded[start + j]
                sumSq += sample * sample
            }
            result[i] = sqrt(sumSq / frameLength)
        }

        return result
    }

    private fun framesToSamples(frame: Int, hopLength: Int): Int {
        return frame * hopLength
    }

    /**
     * Compute RMS energy of audio signal.
     */
    fun computeRms(audio: FloatArray): Float {
        if (audio.isEmpty()) return 0f
        var sumSq = 0f
        for (sample in audio) {
            sumSq += sample * sample
        }
        return sqrt(sumSq / audio.size)
    }
}
