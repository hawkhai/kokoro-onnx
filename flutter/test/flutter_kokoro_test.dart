import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_kokoro/flutter_kokoro.dart';

void main() {
  test('Tokenizer produces correct tokens', () {
    final vocab = {'h': 1, 'e': 2, 'l': 3, 'o': 4};
    final tokenizer = Tokenizer.fromMap(vocab);
    final tokens = tokenizer.tokenize('hello');
    expect(tokens, [1, 2, 3, 3, 4]);
  });

  test('Tokenizer skips unknown characters', () {
    final vocab = {'a': 1, 'b': 2};
    final tokenizer = Tokenizer.fromMap(vocab);
    final tokens = tokenizer.tokenize('aXb');
    expect(tokens, [1, 2]);
  });

  test('AudioProcessor.splitPhonemes splits long text', () {
    final text = 'a' * 600;
    final batches = AudioProcessor.splitPhonemes(text);
    expect(batches.length, greaterThan(1));
    for (final batch in batches) {
      expect(batch.length, lessThanOrEqualTo(KokoroConfig.maxPhonemeLength));
    }
  });

  test('AudioProcessor.splitPhonemes keeps short text as single batch', () {
    final text = 'hello';
    final batches = AudioProcessor.splitPhonemes(text);
    expect(batches.length, 1);
    expect(batches[0], 'hello');
  });

  test('AudioProcessor.trim removes silence', () {
    final audio = Float32List(10000);
    for (int i = 3000; i < 7000; i++) {
      audio[i] = 0.5;
    }
    final trimmed = AudioProcessor.trim(audio);
    expect(trimmed.length, lessThan(audio.length));
  });

  test('AudioProcessor.concat joins audio chunks', () {
    final chunk1 = Float32List.fromList([1.0, 2.0]);
    final chunk2 = Float32List.fromList([3.0, 4.0]);
    final result = AudioProcessor.concat([chunk1, chunk2]);
    expect(result.length, 4);
    expect(result[0], 1.0);
    expect(result[3], 4.0);
  });

  test('VoiceManager.blendVoices mixes correctly', () {
    final v1 = Float32List.fromList([1.0, 0.0, 1.0]);
    final v2 = Float32List.fromList([0.0, 1.0, 0.0]);
    final blended = VoiceManager.blendVoices(v1, v2, 0.5);
    expect(blended[0], 0.5);
    expect(blended[1], 0.5);
    expect(blended[2], 0.5);
  });

  test('KokoroConfig constants are correct', () {
    expect(KokoroConfig.maxPhonemeLength, 510);
    expect(KokoroConfig.sampleRate, 24000);
    expect(KokoroConfig.defaultSpeed, 1.0);
    expect(KokoroConfig.defaultLang, 'en-us');
  });
}
