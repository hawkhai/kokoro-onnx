## 0.1.0

* Initial release of flutter_kokoro plugin.
* Full TTS pipeline: text → phonemization → tokenization → ONNX inference → audio.
* Support for 50+ voices with blending.
* Streaming synthesis via `createStream()`.
* Phonemization via espeak-ng (bundled as native DLL).
* ONNX inference via flutter_onnxruntime.
* Windows platform support.
* Unit tests for tokenizer, audio processor, voice manager.
