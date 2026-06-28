import 'dart:math';
import 'dart:typed_data';

import 'config.dart';

/// Audio processing utilities: silence trimming, batching, concatenation.
///
/// Ported from Python kokoro_onnx/trim.py and __init__.py batch logic.
class AudioProcessor {
  AudioProcessor._();

  /// Trim leading and trailing silence from audio samples.
  ///
  /// Uses RMS energy-based detection with a threshold of [topDb] dB
  /// below the peak RMS. Ported from librosa's effects.trim().
  static Float32List trim(Float32List audio, {double topDb = KokoroConfig.trimTopDb}) {
    if (audio.isEmpty) return audio;

    // Compute frame RMS energy
    const frameLength = 2048;
    const hopLength = 512;
    final nFrames = 1 + (audio.length - frameLength) ~/ hopLength;
    if (nFrames <= 0) return audio;

    final rms = Float32List(nFrames);
    for (int i = 0; i < nFrames; i++) {
      final start = i * hopLength;
      double sum = 0;
      for (int j = 0; j < frameLength && start + j < audio.length; j++) {
        final sample = audio[start + j];
        sum += sample * sample;
      }
      rms[i] = sqrt(sum / frameLength);
    }

    // Find peak RMS
    double peakRms = 0;
    for (int i = 0; i < rms.length; i++) {
      if (rms[i] > peakRms) peakRms = rms[i];
    }

    if (peakRms < KokoroConfig.rmsMin) {
      // All silence
      return Float32List(0);
    }

    // Threshold in linear scale
    final threshold = peakRms * pow(10.0, -topDb / 20.0);

    // Find first non-silent frame
    int startFrame = 0;
    for (int i = 0; i < rms.length; i++) {
      if (rms[i] > threshold) {
        startFrame = i;
        break;
      }
    }

    // Find last non-silent frame
    int endFrame = rms.length - 1;
    for (int i = rms.length - 1; i >= 0; i--) {
      if (rms[i] > threshold) {
        endFrame = i;
        break;
      }
    }

    // Convert frame indices to sample indices
    final startSample = startFrame * hopLength;
    final endSample = min((endFrame + 1) * hopLength + frameLength, audio.length);

    if (startSample >= endSample) return Float32List(0);

    final trimmedLength = endSample - startSample;
    final trimmed = Float32List(trimmedLength);
    for (int i = 0; i < trimmedLength; i++) {
      trimmed[i] = audio[startSample + i];
    }
    return trimmed;
  }

  /// Split phoneme text into batches respecting [maxLen].
  ///
  /// Tries to split at punctuation boundaries for natural-sounding breaks.
  /// Ported from Python Kokoro._split_phonemes().
  static List<String> splitPhonemes(String phonemes, {int maxLen = KokoroConfig.maxPhonemeLength}) {
    if (phonemes.length <= maxLen) {
      return [phonemes];
    }

    final batches = <String>[];
    int start = 0;

    while (start < phonemes.length) {
      if (start + maxLen >= phonemes.length) {
        batches.add(phonemes.substring(start));
        break;
      }

      // Look for a good split point (punctuation or space) near maxLen
      int splitAt = start + maxLen;
      bool found = false;

      // Search backward from maxLen for punctuation
      for (int i = splitAt; i > start + maxLen ~/ 2; i--) {
        final ch = phonemes[i];
        if (ch == ' ' || ch == ',' || ch == '.' || ch == '!' || ch == '?' ||
            ch == ';' || ch == ':' || ch == '-' || ch == '_' ||
            ch == 'ˌ' || ch == 'ˈ') { // IPA stress marks
          splitAt = i + 1;
          found = true;
          break;
        }
      }

      if (!found) {
        // No good split point, just cut at maxLen
        splitAt = start + maxLen;
      }

      batches.add(phonemes.substring(start, splitAt));
      start = splitAt;
    }

    return batches;
  }

  /// Concatenate multiple audio chunks into one.
  static Float32List concat(List<Float32List> chunks) {
    if (chunks.isEmpty) return Float32List(0);
    if (chunks.length == 1) return chunks[0];

    int totalLength = 0;
    for (final chunk in chunks) {
      totalLength += chunk.length;
    }

    final result = Float32List(totalLength);
    int offset = 0;
    for (final chunk in chunks) {
      for (int i = 0; i < chunk.length; i++) {
        result[offset + i] = chunk[i];
      }
      offset += chunk.length;
    }
    return result;
  }
}
