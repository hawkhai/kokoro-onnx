# Kokoro TTS Android

完整的 Android TTS 应用，基于 [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)。

## ✨ 功能

- **完整的独立应用** - 无需依赖其他 APK 或服务
- **自动下载模型** - 首次运行自动从 GitHub 下载模型文件
- **espeak-ng 音素化** - 通过 JNI 集成，支持多语言文本转音素
- **ONNX Runtime 推理** - 本地运行，无需网络
- **多声音支持** - 内置多种声音可选
- **语速调节** - 0.5x 到 2.0x 可调

## 📦 构建

### 前提条件

- Android Studio Arctic Fox 或更高版本
- Android SDK 26+
- NDK (已安装)
- CMake 3.22.1+ (已安装)

### 构建步骤

```bash
cd android
./gradlew assembleDebug
```

生成的 APK: `app/build/outputs/apk/debug/app-debug.apk`

## 🚀 使用

1. **安装 APK** 到 Android 设备或模拟器
2. **首次运行** 会提示下载模型文件 (~380MB)
3. **等待下载完成** (需要网络连接)
4. **输入文本** 并点击 "Generate Speech"
5. **调整参数** (声音、语速)

## 📁 项目结构

```
android/
├── app/src/main/
│   ├── java/com/kokoro/android/
│   │   ├── MainActivity.kt          # 主界面
│   │   ├── Kokoro.kt                # TTS 引擎 (旧版)
│   │   ├── Tokenizer.kt             # 音素 → Token 转换
│   │   ├── AudioTrimmer.kt          # 静音裁剪
│   │   ├── NumpyLoader.kt           # Numpy 文件解析
│   │   ├── WavWriter.kt             # WAV 文件写入
│   │   └── engine/
│   │       ├── KokoroTTSEngine.kt   # 完整 TTS 引擎
│   │       ├── ModelDownloader.kt   # 模型下载管理
│   │       └── EspeakNgEngine.kt    # espeak-ng JNI 包装
│   ├── cpp/
│   │   ├── CMakeLists.txt           # Native 构建配置
│   │   ├── EspeakNgJNI.cpp          # espeak-ng JNI 实现
│   │   └── EspeakNgJNI_stub.cpp     # 无 espeak-ng 时的桩实现
│   └── res/
│       ├── layout/activity_main.xml
│       └── values/
└── scripts/
    └── convert_voices.py            # 声音文件转换工具
```

## 🔧 启用完整音素化 (可选)

默认使用桩实现，音素化功能受限。要启用完整功能：

1. 下载 [espeak-ng 源码](https://github.com/espeak-ng/espeak-ng)
2. 解压到 `app/src/main/cpp/espeak-ng/`
3. 重新构建

```bash
cd app/src/main/cpp
git clone https://github.com/espeak-ng/espeak-ng.git
cd ../../..
./gradlew assembleDebug
```

## 📝 注意事项

- **模型文件大小**: ~380MB，首次下载需要网络
- **支持的架构**: arm64-v8a, armeabi-v7a, x86_64, x86
- **音素化**: 无 espeak-ng 时，需使用预音素化的文本 (IPA)
- **存储权限**: Android 11+ 需要 MANAGE_EXTERNAL_STORAGE

## 📄 许可证

- kokoro-onnx: MIT
- kokoro model: Apache 2.0
- espeak-ng: GPL v3
