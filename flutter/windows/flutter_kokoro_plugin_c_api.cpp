#include "include/flutter_kokoro/flutter_kokoro_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "flutter_kokoro_plugin.h"

void FlutterKokoroPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  flutter_kokoro::FlutterKokoroPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
