import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

/// Direct Dart FFI bindings for espeak-ng.
///
/// Calls espeak-ng shared library directly without a C wrapper.
/// Handles the pointer-to-pointer pattern for espeak_TextToPhonemes.
class EspeakNgBindings {
  late final DynamicLibrary _lib;

  // espeak_Initialize(output, buflength, path, options) -> sample_rate
  late final int Function(int output, int buflength, Pointer<Utf8> path, int options) _initialize;

  // espeak_SetVoiceByName(name) -> error_code
  late final int Function(Pointer<Utf8> name) _setVoiceByName;

  // espeak_TextToPhonemes(textptr, textmode, phonememode) -> phoneme_string
  late final Pointer<Utf8> Function(Pointer<Pointer<Utf8>> textptr, int textmode, int phonememode) _textToPhonemes;

  // espeak_Terminate() -> error_code
  late final int Function() _terminate;

  bool _initialized = false;

  /// Load the espeak-ng shared library.
  void load({String? libraryPath}) {
    if (libraryPath != null) {
      _lib = DynamicLibrary.open(libraryPath);
    } else if (Platform.isWindows) {
      _lib = DynamicLibrary.open('espeak-ng.dll');
    } else if (Platform.isLinux || Platform.isAndroid) {
      _lib = DynamicLibrary.open('libespeak-ng.so');
    } else if (Platform.isMacOS || Platform.isIOS) {
      _lib = DynamicLibrary.open('libespeak-ng.dylib');
    } else {
      throw UnsupportedError('Platform not supported for espeak-ng');
    }

    _initialize = _lib.lookupFunction<
        EspeakInitializeNative,
        EspeakInitializeDart>('espeak_Initialize');

    _setVoiceByName = _lib.lookupFunction<
        EspeakSetVoiceByNameNative,
        EspeakSetVoiceByNameDart>('espeak_SetVoiceByName');

    _textToPhonemes = _lib.lookupFunction<
        EspeakTextToPhonemesNative,
        EspeakTextToPhonemesDart>('espeak_TextToPhonemes');

    _terminate = _lib.lookupFunction<
        EspeakTerminateNative,
        EspeakTerminateDart>('espeak_Terminate');
  }

  /// Initialize espeak-ng.
  ///
  /// [dataPath] - path to espeak-ng-data directory.
  /// Returns sample rate on success (>0), or error code.
  int initialize(String dataPath) {
    final pathPtr = dataPath.toNativeUtf8();
    try {
      // AUDIO_OUTPUT_RETRIEVAL = 1, options = 0
      final result = _initialize(1, 0, pathPtr, 0);
      _initialized = result > 0;
      return result;
    } finally {
      calloc.free(pathPtr);
    }
  }

  /// Set voice by language code (e.g., "en-us", "zh", "ja").
  int setVoiceByName(String name) {
    final namePtr = name.toNativeUtf8();
    try {
      return _setVoiceByName(namePtr);
    } finally {
      calloc.free(namePtr);
    }
  }

  /// Convert text to IPA phonemes.
  ///
  /// Iterates through the text using the pointer-to-pointer pattern.
  String textToPhonemes(String text) {
    if (text.isEmpty) return '';

    final textBytes = text.toNativeUtf8();
    final textPtrPtr = calloc<Pointer<Utf8>>();
    textPtrPtr.value = textBytes;

    // phonememode: 0x02 = IPA, separator '_' (0x5F << 8)
    // Combined: 0x02 | (0x5F << 8) = 0x02 | 0x5F00 = 0x5F02
    const phonemeMode = 0x02 | (0x5F << 8);
    const textMode = 1; // UTF-8

    final result = StringBuffer();

    try {
      while (textPtrPtr.value != nullptr) {
        final phonemePtr = _textToPhonemes(textPtrPtr, textMode, phonemeMode);
        if (phonemePtr != nullptr) {
          result.write(phonemePtr.toDartString());
        }
      }
    } finally {
      calloc.free(textBytes);
      calloc.free(textPtrPtr);
    }

    return result.toString();
  }

  /// Terminate espeak-ng and release resources.
  int terminate() {
    if (!_initialized) return 0;
    _initialized = false;
    return _terminate();
  }

  bool get isInitialized => _initialized;
}

/// High-level phonemizer that wraps EspeakNgBindings.
class Phonemizer {
  final EspeakNgBindings _bindings;

  Phonemizer._(this._bindings);

  /// Create and initialize a Phonemizer.
  ///
  /// [dataPath] - path to espeak-ng-data directory (required).
  /// [libraryPath] - optional explicit path to espeak-ng shared library.
  static Future<Phonemizer> create({
    required String dataPath,
    String? libraryPath,
  }) async {
    final bindings = EspeakNgBindings();
    bindings.load(libraryPath: libraryPath);
    final sampleRate = bindings.initialize(dataPath);
    if (sampleRate <= 0) {
      throw StateError('Failed to initialize espeak-ng (result: $sampleRate)');
    }
    return Phonemizer._(bindings);
  }

  /// Convert text to phonemes for the given language.
  String phonemize(String text, {String lang = 'en-us'}) {
    _bindings.setVoiceByName(lang);
    return _bindings.textToPhonemes(text);
  }

  /// Dispose of espeak-ng resources.
  void dispose() {
    _bindings.terminate();
  }
}

// --- FFI type definitions ---

// espeak_Initialize
typedef EspeakInitializeNative = Int32 Function(
    Int32 output, Int32 buflength, Pointer<Utf8> path, Int32 options);
typedef EspeakInitializeDart = int Function(
    int output, int buflength, Pointer<Utf8> path, int options);

// espeak_SetVoiceByName
typedef EspeakSetVoiceByNameNative = Int32 Function(Pointer<Utf8> name);
typedef EspeakSetVoiceByNameDart = int Function(Pointer<Utf8> name);

// espeak_TextToPhonemes
typedef EspeakTextToPhonemesNative = Pointer<Utf8> Function(
    Pointer<Pointer<Utf8>> textptr, Int32 textmode, Int32 phonememode);
typedef EspeakTextToPhonemesDart = Pointer<Utf8> Function(
    Pointer<Pointer<Utf8>> textptr, int textmode, int phonememode);

// espeak_Terminate
typedef EspeakTerminateNative = Int32 Function();
typedef EspeakTerminateDart = int Function();
