import 'dart:typed_data';

/// Encodes raw PCM float32 audio samples to WAV format.
class WavEncoder {
  WavEncoder._();

  /// Encode float32 PCM samples to WAV bytes.
  ///
  /// [samples] - audio samples in range [-1.0, 1.0]
  /// [sampleRate] - sample rate in Hz (e.g., 24000)
  /// [channels] - number of audio channels (default 1 = mono)
  static Uint8List encode(
    Float32List samples,
    int sampleRate, {
    int channels = 1,
  }) {
    const bitsPerSample = 16;
    final bytesPerSample = bitsPerSample ~/ 8;
    final byteRate = sampleRate * channels * bytesPerSample;
    final blockAlign = channels * bytesPerSample;
    final dataSize = samples.length * bytesPerSample;
    final fileSize = 36 + dataSize;

    final buffer = ByteData(44 + dataSize);
    int offset = 0;

    // RIFF header
    buffer.setUint8(offset++, 0x52); // 'R'
    buffer.setUint8(offset++, 0x49); // 'I'
    buffer.setUint8(offset++, 0x46); // 'F'
    buffer.setUint8(offset++, 0x46); // 'F'
    buffer.setUint32(offset, fileSize, Endian.little);
    offset += 4;
    buffer.setUint8(offset++, 0x57); // 'W'
    buffer.setUint8(offset++, 0x41); // 'A'
    buffer.setUint8(offset++, 0x56); // 'V'
    buffer.setUint8(offset++, 0x45); // 'E'

    // fmt subchunk
    buffer.setUint8(offset++, 0x66); // 'f'
    buffer.setUint8(offset++, 0x6D); // 'm'
    buffer.setUint8(offset++, 0x74); // 't'
    buffer.setUint8(offset++, 0x20); // ' '
    buffer.setUint32(offset, 16, Endian.little); // subchunk size
    offset += 4;
    buffer.setUint16(offset, 1, Endian.little); // PCM format
    offset += 2;
    buffer.setUint16(offset, channels, Endian.little);
    offset += 2;
    buffer.setUint32(offset, sampleRate, Endian.little);
    offset += 4;
    buffer.setUint32(offset, byteRate, Endian.little);
    offset += 4;
    buffer.setUint16(offset, blockAlign, Endian.little);
    offset += 2;
    buffer.setUint16(offset, bitsPerSample, Endian.little);
    offset += 2;

    // data subchunk
    buffer.setUint8(offset++, 0x64); // 'd'
    buffer.setUint8(offset++, 0x61); // 'a'
    buffer.setUint8(offset++, 0x74); // 't'
    buffer.setUint8(offset++, 0x61); // 'a'
    buffer.setUint32(offset, dataSize, Endian.little);
    offset += 4;

    // Convert float32 samples to int16 PCM
    for (int i = 0; i < samples.length; i++) {
      final sample = samples[i].clamp(-1.0, 1.0);
      final intSample = (sample * 32767).round();
      buffer.setInt16(offset, intSample, Endian.little);
      offset += 2;
    }

    return buffer.buffer.asUint8List();
  }
}
