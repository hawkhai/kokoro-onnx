#ifndef FLUTTER_PLUGIN_FLUTTER_KOKORO_PLUGIN_H_
#define FLUTTER_PLUGIN_FLUTTER_KOKORO_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace flutter_kokoro {

class FlutterKokoroPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows *registrar);

  FlutterKokoroPlugin();

  virtual ~FlutterKokoroPlugin();

  // Disallow copy and assign.
  FlutterKokoroPlugin(const FlutterKokoroPlugin&) = delete;
  FlutterKokoroPlugin& operator=(const FlutterKokoroPlugin&) = delete;

  // Called when a method is called on this plugin's channel from Dart.
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue> &method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
};

}  // namespace flutter_kokoro

#endif  // FLUTTER_PLUGIN_FLUTTER_KOKORO_PLUGIN_H_
