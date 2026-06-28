import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_kokoro/flutter_kokoro.dart';
import 'package:path/path.dart' as p;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final _kokoro = KokoroTts();
  final _textController = TextEditingController(text: 'Hello, this is a test of Kokoro text to speech.');
  String _status = 'Not initialized';
  List<String> _voices = [];
  String _selectedVoice = '';

  @override
  void initState() {
    super.initState();
    _initEngine();
  }

  Future<void> _initEngine() async {
    try {
      setState(() => _status = 'Initializing...');

      // Paths relative to the executable location.
      // Model files should be placed in the models/ directory next to the exe.
      final exeDir = p.dirname(Platform.resolvedExecutable);
      final modelPath = p.join(exeDir, 'models', 'kokoro-v1.0.onnx');
      final voicesPath = p.join(exeDir, 'models', 'voices-v1.0.bin');
      final espeakDataPath = p.join(exeDir, 'data', 'espeak-ng-data');

      // Verify files exist
      if (!File(modelPath).existsSync()) {
        setState(() {
          _status = 'Model not found at $modelPath\n'
              'Please copy kokoro-v1.0.onnx to the models/ directory next to the executable.';
        });
        return;
      }
      if (!File(voicesPath).existsSync()) {
        setState(() {
          _status = 'Voices not found at $voicesPath\n'
              'Please copy voices-v1.0.bin to the models/ directory next to the executable.';
        });
        return;
      }
      if (!Directory(espeakDataPath).existsSync()) {
        setState(() {
          _status = 'espeak-ng-data not found at $espeakDataPath';
        });
        return;
      }

      await _kokoro.init(
        modelPath: modelPath,
        voicesPath: voicesPath,
        espeakDataPath: espeakDataPath,
      );

      final voices = _kokoro.getVoices();
      setState(() {
        _voices = voices;
        _selectedVoice = voices.isNotEmpty ? voices.first : '';
        _status = 'Ready (${voices.length} voices available)';
      });
    } catch (e) {
      setState(() => _status = 'Error: $e');
    }
  }

  Future<void> _synthesize() async {
    if (!_kokoro.isReady) return;
    try {
      setState(() => _status = 'Synthesizing...');
      final stopwatch = Stopwatch()..start();
      final (audio, sampleRate) = await _kokoro.create(
        text: _textController.text,
        voice: _selectedVoice,
      );
      stopwatch.stop();
      final durationSec = audio.length / sampleRate;
      setState(() {
        _status = 'Generated ${audio.length} samples '
            '(${durationSec.toStringAsFixed(1)}s audio) '
            'at ${sampleRate}Hz in ${stopwatch.elapsedMilliseconds}ms';
      });
      // TODO: Play audio using a player package (e.g., just_audio)
    } catch (e) {
      setState(() => _status = 'Error: $e');
    }
  }

  @override
  void dispose() {
    _textController.dispose();
    _kokoro.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Kokoro TTS Example')),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text('Status: $_status', style: Theme.of(context).textTheme.bodyMedium),
              const SizedBox(height: 16),
              TextField(
                controller: _textController,
                maxLines: 3,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  labelText: 'Text to synthesize',
                ),
              ),
              const SizedBox(height: 16),
              if (_voices.isNotEmpty) ...[
                DropdownButtonFormField<String>(
                  initialValue: _selectedVoice,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    labelText: 'Voice',
                  ),
                  items: _voices
                      .map((v) => DropdownMenuItem(value: v, child: Text(v)))
                      .toList(),
                  onChanged: (v) => setState(() => _selectedVoice = v ?? ''),
                ),
                const SizedBox(height: 16),
                ElevatedButton.icon(
                  onPressed: _synthesize,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Synthesize'),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
