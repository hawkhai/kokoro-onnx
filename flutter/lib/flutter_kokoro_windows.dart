/// Windows platform implementation for flutter_kokoro.
///
/// This class is registered via dartPluginClass in pubspec.yaml.
/// Since we use dart:ffi for all native code, this is a minimal stub
/// that satisfies the Flutter plugin registration requirement.
class FlutterKokoroWindows {
  /// Registers this plugin with the Flutter engine.
  static void registerWith() {
    // No-op: all native code is accessed via dart:ffi directly.
    // This method exists to satisfy the dartPluginClass requirement.
  }
}
