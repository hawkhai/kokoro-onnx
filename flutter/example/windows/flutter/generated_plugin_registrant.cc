//
//  Generated file. Do not edit.
//

// clang-format off

#include "generated_plugin_registrant.h"

#include <flutter_kokoro/flutter_kokoro_plugin_c_api.h>
#include <flutter_onnxruntime/flutter_onnxruntime_plugin.h>

void RegisterPlugins(flutter::PluginRegistry* registry) {
  FlutterKokoroPluginCApiRegisterWithRegistrar(
      registry->GetRegistrarForPlugin("FlutterKokoroPluginCApi"));
  FlutterOnnxruntimePluginRegisterWithRegistrar(
      registry->GetRegistrarForPlugin("FlutterOnnxruntimePlugin"));
}
