/// Constants for Kokoro TTS.
class KokoroConfig {
  KokoroConfig._();

  /// Maximum phoneme length per inference batch.
  static const int maxPhonemeLength = 510;

  /// Audio sample rate in Hz.
  static const int sampleRate = 24000;

  /// Minimum allowed synthesis speed.
  static const double minSpeed = 0.5;

  /// Maximum allowed synthesis speed.
  static const double maxSpeed = 2.0;

  /// Default synthesis speed.
  static const double defaultSpeed = 1.0;

  /// Default language.
  static const String defaultLang = 'en-us';

  /// RMS silence threshold for audio trimming (in dB).
  static const double trimTopDb = 60.0;

  /// Minimum RMS energy threshold to avoid division by zero.
  static const double rmsMin = 1e-10;
}
