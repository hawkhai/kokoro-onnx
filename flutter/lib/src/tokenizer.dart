import 'dart:convert';

/// Tokenizer that converts phoneme strings to integer token IDs.
///
/// Ported from Python kokoro_onnx/tokenizer.py
class Tokenizer {
  final Map<String, int> _vocab;

  Tokenizer(this._vocab);

  /// Create a tokenizer from a JSON config string (the config.json content).
  factory Tokenizer.fromJson(String jsonStr) {
    final Map<String, dynamic> config = jsonDecode(jsonStr);
    final Map<String, dynamic> vocabMap = config['vocab'] ?? config;
    final vocab = vocabMap.map((k, v) => MapEntry(k, v as int));
    return Tokenizer(vocab);
  }

  /// Create a tokenizer from a vocab map.
  factory Tokenizer.fromMap(Map<String, int> vocab) {
    return Tokenizer(vocab);
  }

  /// Convert a phoneme string to a list of token IDs.
  ///
  /// Each phoneme character is looked up in the vocabulary.
  /// Characters not in the vocab are silently skipped.
  List<int> tokenize(String phonemes) {
    final tokens = <int>[];
    for (final char in phonemes.split('')) {
      final id = _vocab[char];
      if (id != null) {
        tokens.add(id);
      }
    }
    return tokens;
  }

  /// Get the vocabulary map.
  Map<String, int> get vocab => Map.unmodifiable(_vocab);
}
