/// Flutter plugin for Kokoro TTS - text-to-speech using Kokoro-82M ONNX model.
///
/// This plugin provides a complete TTS pipeline:
/// - Text phonemization (via espeak-ng)
/// - Phoneme tokenization
/// - ONNX model inference
/// - Audio post-processing (silence trimming, batching)
///
/// ## Quick start
///
/// ```dart
/// import 'package:flutter_kokoro/flutter_kokoro.dart';
///
/// final kokoro = KokoroTts();
/// await kokoro.init(
///   modelPath: '/path/to/kokoro-v1.0.onnx',
///   voicesPath: '/path/to/voices-v1.0.bin',
///   espeakDataPath: '/path/to/espeak-ng-data',
/// );
///
/// final voices = kokoro.getVoices();
/// final (audio, sampleRate) = await kokoro.create(
///   text: 'Hello world',
///   voice: 'af_sarah',
/// );
///
/// kokoro.dispose();
/// ```
library;

export 'src/kokoro_tts.dart' show KokoroTts, AudioResult;
export 'src/config.dart' show KokoroConfig;
export 'src/voice_manager.dart' show VoiceManager;
export 'src/phonemizer.dart' show Phonemizer;
export 'src/tokenizer.dart' show Tokenizer;
export 'src/audio_processor.dart' show AudioProcessor;
export 'src/npz_parser.dart' show NpzParser, NpyArray;
export 'flutter_kokoro_windows.dart' show FlutterKokoroWindows;
