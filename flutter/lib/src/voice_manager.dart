import 'dart:typed_data';

import 'npz_parser.dart';

/// Manages Kokoro voice style embeddings.
///
/// Voice files (.bin) are numpy .npz archives containing per-voice float32
/// arrays. Each voice is a 2D array indexed by phoneme sequence length.
class VoiceManager {
  final Map<String, Float32List> _voices;
  final Map<String, List<int>> _shapes;

  VoiceManager._(this._voices, this._shapes);

  /// Load voices from a .bin (npz) file.
  static Future<VoiceManager> load(String path) async {
    final arrays = await NpzParser.loadWithShapes(path);
    final voices = <String, Float32List>{};
    final shapes = <String, List<int>>{};
    for (final entry in arrays.entries) {
      voices[entry.key] = entry.value.data;
      shapes[entry.key] = entry.value.shape;
    }
    return VoiceManager._(voices, shapes);
  }

  /// Get a voice style array by name.
  ///
  /// Returns null if the voice is not found.
  Float32List? getVoice(String name) => _voices[name];

  /// Get the shape of a voice style array.
  List<int>? getShape(String name) => _shapes[name];

  /// Get all available voice names, sorted alphabetically.
  List<String> getVoiceNames() {
    final names = _voices.keys.toList()..sort();
    return names;
  }

  /// Check if a voice exists.
  bool hasVoice(String name) => _voices.containsKey(name);

  /// Blend two voice styles with the given ratio.
  ///
  /// [ratio] controls the mix: 0.0 = all voice1, 1.0 = all voice2.
  /// Both voices must have the same shape.
  static Float32List blendVoices(
    Float32List voice1,
    Float32List voice2,
    double ratio,
  ) {
    if (voice1.length != voice2.length) {
      throw ArgumentError(
        'Voice arrays must have the same length '
        '(${voice1.length} vs ${voice2.length})',
      );
    }
    if (ratio < 0.0 || ratio > 1.0) {
      throw ArgumentError('Ratio must be between 0.0 and 1.0');
    }

    final result = Float32List(voice1.length);
    for (int i = 0; i < voice1.length; i++) {
      result[i] = voice1[i] * (1.0 - ratio) + voice2[i] * ratio;
    }
    return result;
  }
}
