# flutter_kokoro

Flutter plugin for [Kokoro TTS](https://github.com/thewh1teagle/kokoro-onnx) — text-to-speech using the Kokoro-82M ONNX model.

## Features

- **Full TTS pipeline**: text → phonemization → tokenization → ONNX inference → audio
- **Multiple voices**: 50+ voices with blending support
- **Streaming**: generate audio chunks incrementally for real-time playback
- **Multi-language**: English, Chinese, Japanese, and more via espeak-ng
- **Cross-platform**: Windows (Android, iOS, Linux, macOS planned)

## Requirements

- Flutter 3.3.0+
- Model files (see [Setup](#setup))

## Setup

### 1. Add dependency

```yaml
dependencies:
  flutter_kokoro:
    path: ../path/to/flutter_kokoro
```

### 2. Download model files

Download from [Kokoro-82M releases](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0):

- [`kokoro-v1.0.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx) (~310MB)
- [`voices-v1.0.bin`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin) (~27MB)

Place them in a `models/` directory next to your executable.

### 3. Bundle espeak-ng data

The plugin bundles `espeak-ng.dll` automatically. The `espeak-ng-data` directory is also copied to the build output's `data/` folder.

## Usage

```dart
import 'package:flutter_kokoro/flutter_kokoro.dart';

// Initialize
final kokoro = KokoroTts();
await kokoro.init(
  modelPath: 'models/kokoro-v1.0.onnx',
  voicesPath: 'models/voices-v1.0.bin',
  espeakDataPath: 'data/espeak-ng-data',
);

// List available voices
final voices = kokoro.getVoices();
// ['af_heart', 'af_sarah', 'am_michael', ...]

// Synthesize speech
final (audio, sampleRate) = await kokoro.create(
  text: 'Hello, world!',
  voice: 'af_sarah',
  speed: 1.0,
  lang: 'en-us',
);
// audio: Float32List, sampleRate: 24000

// Stream synthesis (for long text)
await for (final (chunk, sr) in kokoro.createStream(
  text: longText,
  voice: 'af_sarah',
)) {
  // Play each chunk as it arrives
}

// Blend voices
final style1 = kokoro.getVoiceStyle('af_sarah');
final style2 = kokoro.getVoiceStyle('am_michael');
final blended = KokoroTts.blendVoices(style1!, style2!, 0.5);
final (audio, sr) = await kokoro.create(text: 'Hi', voice: blended);

// Clean up
kokoro.dispose();
```

## API Reference

### `KokoroTts`

| Method | Description |
|--------|-------------|
| `init(modelPath, voicesPath, espeakDataPath)` | Initialize the TTS engine |
| `create(text, voice, {speed, lang, isPhonemes, trim})` | Synthesize audio (returns `Float32List` + sample rate) |
| `createStream(text, voice, ...)` | Stream audio chunks via `Stream` |
| `getVoices()` | Get list of available voice names |
| `getVoiceStyle(name)` | Get raw voice style array for blending |
| `blendVoices(v1, v2, ratio)` | Static method to blend two voice styles |
| `dispose()` | Release all resources |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `String` | required | Text to synthesize |
| `voice` | `String` or `Float32List` | required | Voice name or blended style |
| `speed` | `double` | `1.0` | Synthesis speed (0.5–2.0) |
| `lang` | `String` | `'en-us'` | Language for phonemization |
| `isPhonemes` | `bool` | `false` | Treat text as pre-phonemized |
| `trim` | `bool` | `true` | Trim silence from audio chunks |

## Architecture

```
Text → espeak-ng (phonemization) → Tokenizer → ONNX Runtime → Audio
         ↓                              ↓            ↓
    Phonemizer                    config.json    flutter_onnxruntime
    (dart:ffi)                    (vocab map)    (inference)
```

## License

MIT (same as kokoro-onnx)
