import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

import 'audio_processor.dart';
import 'config.dart';
import 'phonemizer.dart';
import 'tokenizer.dart';
import 'voice_manager.dart';

/// Result of TTS synthesis: audio samples and sample rate.
typedef AudioResult = (Float32List audio, int sampleRate);

/// Kokoro TTS engine.
///
/// Wraps the full text-to-speech pipeline: phonemization, tokenization,
/// ONNX inference, and audio post-processing.
///
/// Usage:
/// ```dart
/// final kokoro = KokoroTts();
/// await kokoro.init(
///   modelPath: 'path/to/kokoro-v1.0.onnx',
///   voicesPath: 'path/to/voices-v1.0.bin',
///   espeakDataPath: 'path/to/espeak-ng-data',
/// );
/// final (audio, sampleRate) = await kokoro.create(text: 'Hello', voice: 'af_sarah');
/// kokoro.dispose();
/// ```
class KokoroTts {
  OnnxRuntime? _ort;
  OrtSession? _session;
  Phonemizer? _phonemizer;
  Tokenizer? _tokenizer;
  VoiceManager? _voiceManager;

  bool _initialized = false;

  /// Initialize the TTS engine.
  ///
  /// [modelPath] - path to the ONNX model file.
  /// [voicesPath] - path to the voices .bin (npz) file.
  /// [espeakDataPath] - path to espeak-ng-data directory.
  /// [vocabConfig] - optional custom vocab JSON string or map.
  Future<void> init({
    required String modelPath,
    required String voicesPath,
    required String espeakDataPath,
    dynamic vocabConfig,
  }) async {
    // Load config.json for vocabulary
    Map<String, int> vocab;
    if (vocabConfig is Map) {
      vocab = vocabConfig.map((k, v) => MapEntry(k, v as int));
    } else if (vocabConfig is String) {
      final config = jsonDecode(vocabConfig) as Map<String, dynamic>;
      vocab = (config['vocab'] as Map<String, dynamic>? ?? config)
          .map((k, v) => MapEntry(k, v as int));
    } else {
      // Load default config.json from assets
      final configStr = await rootBundle.loadString(
        'packages/flutter_kokoro/assets/config.json',
      );
      final config = jsonDecode(configStr) as Map<String, dynamic>;
      vocab = (config['vocab'] as Map<String, dynamic>? ?? config)
          .map((k, v) => MapEntry(k, v as int));
    }

    // Initialize components
    _tokenizer = Tokenizer.fromMap(vocab);
    _phonemizer = await Phonemizer.create(dataPath: espeakDataPath);
    _voiceManager = await VoiceManager.load(voicesPath);

    // Initialize ONNX Runtime
    _ort = OnnxRuntime();
    _session = await _ort!.createSession(modelPath);

    _initialized = true;
  }

  /// Initialize from asset paths (files bundled in the app's assets).
  ///
  /// This loads model and voices from the asset bundle.
  /// Note: large model files should be loaded from the filesystem, not assets.
  Future<void> initFromAssets({
    required String modelAssetPath,
    required String voicesAssetPath,
    required String espeakDataPath,
  }) async {
    // For asset-based init, we'd need to copy assets to temp files first
    // since ONNX Runtime needs file paths. This is a convenience method.
    throw UnimplementedError(
      'Asset-based init not yet implemented. Use init() with file paths.',
    );
  }

  /// Whether the engine is initialized and ready.
  bool get isReady => _initialized;

  /// Get all available voice names.
  List<String> getVoices() {
    _ensureInitialized();
    return _voiceManager!.getVoiceNames();
  }

  /// Get a voice style array by name.
  Float32List? getVoiceStyle(String name) {
    _ensureInitialized();
    return _voiceManager!.getVoice(name);
  }

  /// Blend two voice styles.
  ///
  /// [ratio] controls the mix: 0.0 = all voice1, 1.0 = all voice2.
  static Float32List blendVoices(
    Float32List voice1,
    Float32List voice2,
    double ratio,
  ) {
    return VoiceManager.blendVoices(voice1, voice2, ratio);
  }

  /// Synthesize audio from text.
  ///
  /// [text] - input text to synthesize.
  /// [voice] - voice name (e.g., "af_sarah") or a pre-blended Float32List.
  /// [speed] - synthesis speed, 0.5 to 2.0.
  /// [lang] - language code for phonemization.
  /// [isPhonemes] - if true, treat [text] as pre-phonemized text.
  /// [trim] - whether to trim silence from audio chunks.
  ///
  /// Returns a tuple of (audio samples as Float32List, sample rate 24000).
  Future<AudioResult> create({
    required String text,
    required dynamic voice,
    double speed = KokoroConfig.defaultSpeed,
    String lang = KokoroConfig.defaultLang,
    bool isPhonemes = false,
    bool trim = true,
  }) async {
    _ensureInitialized();

    if (speed < KokoroConfig.minSpeed || speed > KokoroConfig.maxSpeed) {
      throw ArgumentError(
        'Speed must be between ${KokoroConfig.minSpeed} and ${KokoroConfig.maxSpeed}',
      );
    }

    // Resolve voice style
    final Float32List voiceStyle;
    if (voice is String) {
      final style = _voiceManager!.getVoice(voice);
      if (style == null) {
        throw ArgumentError('Voice "$voice" not found');
      }
      voiceStyle = style;
    } else if (voice is Float32List) {
      voiceStyle = voice;
    } else {
      throw ArgumentError('Voice must be a name (String) or Float32List');
    }

    // Phonemize
    String phonemes;
    if (isPhonemes) {
      phonemes = text;
    } else {
      final rawPhonemes = _phonemizer!.phonemize(text, lang: lang);
      // Filter to known phonemes
      final filtered = StringBuffer();
      for (final ch in rawPhonemes.split('')) {
        if (_tokenizer!.vocab.containsKey(ch)) {
          filtered.write(ch);
        }
      }
      phonemes = filtered.toString().trim();
    }

    if (phonemes.isEmpty) {
      return (Float32List(0), KokoroConfig.sampleRate);
    }

    // Split into batches
    final batches = AudioProcessor.splitPhonemes(phonemes);

    // Run inference on each batch
    final audioChunks = <Float32List>[];
    for (final batch in batches) {
      final chunk = await _inferBatch(batch, voiceStyle, speed, trim);
      if (chunk.isNotEmpty) {
        audioChunks.add(chunk);
      }
    }

    // Concatenate all chunks
    final audio = AudioProcessor.concat(audioChunks);
    return (audio, KokoroConfig.sampleRate);
  }

  /// Stream audio chunks as they are generated.
  ///
  /// Same parameters as [create], but yields audio chunks one at a time.
  /// Useful for long text where you want to start playback before synthesis
  /// is complete.
  Stream<AudioResult> createStream({
    required String text,
    required dynamic voice,
    double speed = KokoroConfig.defaultSpeed,
    String lang = KokoroConfig.defaultLang,
    bool isPhonemes = false,
    bool trim = true,
  }) async* {
    _ensureInitialized();

    if (speed < KokoroConfig.minSpeed || speed > KokoroConfig.maxSpeed) {
      throw ArgumentError(
        'Speed must be between ${KokoroConfig.minSpeed} and ${KokoroConfig.maxSpeed}',
      );
    }

    // Resolve voice style
    final Float32List voiceStyle;
    if (voice is String) {
      final style = _voiceManager!.getVoice(voice);
      if (style == null) {
        throw ArgumentError('Voice "$voice" not found');
      }
      voiceStyle = style;
    } else if (voice is Float32List) {
      voiceStyle = voice;
    } else {
      throw ArgumentError('Voice must be a name (String) or Float32List');
    }

    // Phonemize
    String phonemes;
    if (isPhonemes) {
      phonemes = text;
    } else {
      final rawPhonemes = _phonemizer!.phonemize(text, lang: lang);
      final filtered = StringBuffer();
      for (final ch in rawPhonemes.split('')) {
        if (_tokenizer!.vocab.containsKey(ch)) {
          filtered.write(ch);
        }
      }
      phonemes = filtered.toString().trim();
    }

    if (phonemes.isEmpty) return;

    // Split and yield each batch
    final batches = AudioProcessor.splitPhonemes(phonemes);
    for (final batch in batches) {
      final chunk = await _inferBatch(batch, voiceStyle, speed, trim);
      if (chunk.isNotEmpty) {
        yield (chunk, KokoroConfig.sampleRate);
      }
    }
  }

  /// Run ONNX inference on a single phoneme batch.
  Future<Float32List> _inferBatch(
    String phonemes,
    Float32List voiceStyle,
    double speed,
    bool trim,
  ) async {
    // Tokenize
    final tokens = _tokenizer!.tokenize(phonemes);
    if (tokens.isEmpty) return Float32List(0);

    // Select voice style by token count.
    // Voice array shape is (510, 1, 256) flattened to 130560 elements.
    // Each phoneme length has a 256-dim style vector.
    // Layout: element[length][0][j] = flat[length * 256 + j]
    const styleDim = 256;
    final tokenCount = tokens.length;
    final styleOffset = tokenCount * styleDim;

    // Bounds check
    if (styleOffset + styleDim > voiceStyle.length) {
      // Fallback: use the last available style
      final fallbackOffset = voiceStyle.length - styleDim;
      final styleSlice = Float32List(styleDim);
      for (int i = 0; i < styleDim; i++) {
        styleSlice[i] = voiceStyle[fallbackOffset + i];
      }
      return _runOnnx(tokens, styleSlice, speed, trim);
    }

    final styleSlice = Float32List(styleDim);
    for (int i = 0; i < styleDim; i++) {
      styleSlice[i] = voiceStyle[styleOffset + i];
    }

    return _runOnnx(tokens, styleSlice, speed, trim);
  }

  /// Execute the ONNX model.
  Future<Float32List> _runOnnx(
    List<int> tokens,
    Float32List style,
    double speed,
    bool trim,
  ) async {
    // Pad tokens with 0 at start and end, convert to Int64List (model requires int64)
    final paddedTokens = Int64List.fromList([0, ...tokens, 0]);

    // Create input tensors
    final inputIds = await OrtValue.fromList(paddedTokens, [1, paddedTokens.length]);
    final styleTensor = await OrtValue.fromList(style.toList(), [1, style.length]);
    final speedTensor = await OrtValue.fromList(Float32List.fromList([speed]), [1]);

    try {
      // Determine input names (handle both old and new model formats)
      final inputNames = _session!.inputNames;
      final tokenIdName = inputNames.contains('input_ids') ? 'input_ids' : 'tokens';

      final inputs = {
        tokenIdName: inputIds,
        'style': styleTensor,
        'speed': speedTensor,
      };

      final outputs = await _session!.run(inputs);

      // Get the output audio
      final outputKey = outputs.keys.first;
      final outputValue = outputs[outputKey]!;
      final audioList = await outputValue.asList();

      // Convert to Float32List
      final audio = Float32List(audioList.length);
      for (int i = 0; i < audioList.length; i++) {
        audio[i] = (audioList[i] as num).toDouble();
      }

      // Trim silence
      if (trim) {
        return AudioProcessor.trim(audio);
      }
      return audio;
    } finally {
      await inputIds.dispose();
      await styleTensor.dispose();
      await speedTensor.dispose();
    }
  }

  void _ensureInitialized() {
    if (!_initialized) {
      throw StateError('KokoroTts not initialized. Call init() first.');
    }
  }

  /// Release all resources.
  Future<void> dispose() async {
    _phonemizer?.dispose();
    await _session?.close();
    _initialized = false;
  }
}
