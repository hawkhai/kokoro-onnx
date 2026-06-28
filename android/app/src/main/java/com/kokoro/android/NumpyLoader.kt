package com.kokoro.android

import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Utility for loading numpy .npy files on Android.
 *
 * Supports loading numpy arrays saved with np.save().
 * For dictionary-like .npz files, use loadNpz().
 */
object NumpyLoader {

    /**
     * Load a numpy .npy file as a FloatArray.
     * Supports 1D and 2D float32 arrays.
     */
    fun loadFloatArray(filePath: String): FloatArray {
        val file = File(filePath)
        val bytes = file.readBytes()
        return parseNpyFloatArray(bytes)
    }

    /**
     * Load a numpy .npy file as a 2D FloatArray (array of arrays).
     */
    fun load2DFloatArray(filePath: String): Array<FloatArray> {
        val file = File(filePath)
        val bytes = file.readBytes()
        return parseNpy2DFloatArray(bytes)
    }

    /**
     * Parse numpy .npy format bytes into a 1D FloatArray.
     */
    private fun parseNpyFloatArray(data: ByteArray): FloatArray {
        val header = parseNpyHeader(data)
        val dataOffset = header.dataOffset

        if (header.descr != "<f4" && header.descr != "<f8") {
            throw IllegalArgumentException("Unsupported dtype: ${header.descr}. Only float32 and float64 are supported.")
        }

        val numElements = header.shape.fold(1) { acc, dim -> acc * dim }
        val buffer = ByteBuffer.wrap(data, dataOffset, data.size - dataOffset)
            .order(ByteOrder.LITTLE_ENDIAN)

        return FloatArray(numElements) {
            if (header.descr == "<f4") buffer.float else buffer.double.toFloat()
        }
    }

    /**
     * Parse numpy .npy format bytes into a 2D FloatArray.
     */
    private fun parseNpy2DFloatArray(data: ByteArray): Array<FloatArray> {
        val header = parseNpyHeader(data)
        val dataOffset = header.dataOffset

        if (header.shape.size != 2) {
            throw IllegalArgumentException("Expected 2D array, got ${header.shape.size}D")
        }

        val rows = header.shape[0]
        val cols = header.shape[1]
        val buffer = ByteBuffer.wrap(data, dataOffset, data.size - dataOffset)
            .order(ByteOrder.LITTLE_ENDIAN)

        return Array(rows) {
            FloatArray(cols) {
                if (header.descr == "<f4") buffer.float else buffer.double.toFloat()
            }
        }
    }

    /**
     * Parse the .npy file header.
     */
    private fun parseNpyHeader(data: ByteArray): NpyHeader {
        // Magic: \x93NUMPY
        if (data[0] != 0x93.toByte() ||
            String(data, 1, 5) != "NUMPY"
        ) {
            throw IllegalArgumentException("Not a valid .npy file")
        }

        val majorVersion = data[6].toInt() and 0xFF
        val headerLen: Int
        val headerStart: Int

        if (majorVersion == 1) {
            headerLen = ByteBuffer.wrap(data, 8, 2)
                .order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 0xFFFF
            headerStart = 10
        } else {
            headerLen = ByteBuffer.wrap(data, 8, 4)
                .order(ByteOrder.LITTLE_ENDIAN).int
            headerStart = 12
        }

        val headerStr = String(data, headerStart, headerLen).trim()

        // Parse descr
        val descrMatch = Regex("'descr':\\s*'([^']+)'").find(headerStr)
            ?: throw IllegalArgumentException("Cannot parse descr from header")
        val descr = descrMatch.groupValues[1]

        // Parse shape
        val shapeMatch = Regex("'shape':\\s*\\(([^)]*)\\)").find(headerStr)
            ?: throw IllegalArgumentException("Cannot parse shape from header")
        val shapeStr = shapeMatch.groupValues[1].trim()
        val shape = if (shapeStr.isEmpty()) {
            intArrayOf()
        } else {
            shapeStr.split(",").filter { it.isNotBlank() }.map { it.trim().toInt() }.toIntArray()
        }

        // Parse fortran_order
        val fortranMatch = Regex("'fortran_order':\\s*(True|False)").find(headerStr)
        val fortranOrder = fortranMatch?.groupValues?.get(1) == "True"

        val dataOffset = headerStart + headerLen

        return NpyHeader(descr, shape, fortranOrder, dataOffset)
    }

    /**
     * Load voices from a .npz file (numpy zip archive).
     * Returns a map of voice name -> 2D float array (token_length x style_dim).
     */
    fun loadVoices(filePath: String): Map<String, Array<FloatArray>> {
        val voices = mutableMapOf<String, Array<FloatArray>>()

        java.util.zip.ZipFile(File(filePath)).use { zip ->
            val entries = zip.entries()
            while (entries.hasMoreElements()) {
                val entry = entries.nextElement()
                if (entry.name.endsWith(".npy")) {
                    val voiceName = entry.name.removeSuffix(".npy")
                    val bytes = zip.getInputStream(entry).readBytes()
                    voices[voiceName] = parseNpy2DFloatArray(bytes)
                }
            }
        }

        return voices
    }

    data class NpyHeader(
        val descr: String,
        val shape: IntArray,
        val fortranOrder: Boolean,
        val dataOffset: Int
    )
}
