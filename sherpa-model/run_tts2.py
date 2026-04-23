import time
import sherpa_onnx
REPO = "F:/pythonx/mytts/kokoro-onnx/sherpa-model-repo"
kokoro_config = sherpa_onnx.OfflineTtsKokoroModelConfig(
    model=REPO + "/model.onnx",
    voices=REPO + "/voices.bin",
    tokens=REPO + "/tokens.txt",
    lexicon=REPO + "/lexicon-zh.txt",
    data_dir=REPO + "/espeak-ng-data",
    dict_dir=REPO + "/dict",
    length_scale=1.0,
)
model_config = sherpa_onnx.OfflineTtsModelConfig(
    kokoro=kokoro_config,
    num_threads=4,
    debug=True,
    provider="cpu",
)
rule_fsts = ",".join([
    REPO + "/date-zh.fst",
    REPO + "/number-zh.fst",
    REPO + "/phone-zh.fst",
])
tts_config = sherpa_onnx.OfflineTtsConfig(
    model=model_config,
    rule_fsts=rule_fsts,
    max_num_sentences=1,
)
print("Validating config...", flush=True)
if not tts_config.validate():
    raise RuntimeError("TTS config validation failed!")
print("Creating TTS engine...", flush=True)
tts = sherpa_onnx.OfflineTts(tts_config)
print("Engine ready!", flush=True)
text = r"""


领取优惠
https://github.com/k2-fsa/sherpa-ncnn  这里里面最推荐的 文本转语音 是哪个方案？

结论先说清楚：
👉 sherpa-ncnn 项目 里面“最推荐”的 TTS 其实只有一类核心方案——👉基于 Piper 的 VITS 模型，没有第二梯队可选。

一、sherpa-ncnn 的 TTS 本质
sherpa-ncnn 本身 不是一个 TTS 模型集合，而是一个推理框架，它的 TTS：

✔ 实际上就是 运行 Piper 的 VITS 模型（转成 ncnn） 

也就是说：

sherpa-ncnn = 推理框架（ncnn）

TTS能力 = Piper VITS 模型导出版本

👉 所以你在这个 repo 里“选方案”，本质是在选：
哪一个 Piper VITS 模型更适合你

二、当前唯一主线：Piper (VITS)
从官方 release 也能看出来：

新版本持续在做：

支持 multi-speaker piper

导出更多 piper 模型 

👉 说明路线已经定死了：

✅ 官方唯一推荐 TTS：Piper (VITS)

三、那“最推荐模型”具体选哪个？
重点来了（这才是你真正想问的）

1️⃣ 英文（最成熟）
推荐优先级：

⭐ en_US-lessac-medium

⭐ en_GB-semaine-medium

👉 特点：

体积：50MB~100MB

速度：极快（RTF < 0.2，接近实时甚至更快）

音质：中上（但不如 Kokoro）

👉 Reddit 实测也类似：

“RTF 0.117，接近 9x realtime” 

2️⃣ 中文（重点）
现实一点说：

👉 Piper 中文 = 能用，但不算强

推荐：

⭐ zh_CN-huayan / zh_CN-xiaoyi（常见社区模型）

但问题：

断句差（你已经遇到了）

prosody 很弱

偏“朗读机器人”

3️⃣ 多说话人（新版本支持）
如果你想要“更高级一点”：

multi-speaker piper（新版本支持） 

👉 优点：

可以切换 speaker

更接近“产品级”

👉 但：

本质还是 VITS → 上限有限

四、为什么没有更好的？
因为 sherpa-ncnn 的设计目标不是效果，而是：

✔ 极致部署能力
无 PyTorch

无 ONNX Runtime

纯 ncnn

可跑：

手机

树莓派

RISC-V

👉 所以它选了：

❗ 最轻量稳定的方案（Piper VITS），而不是最强模型

五、如果你要“最优解”，真实建议
我直接给你工业级结论（不绕弯）：

❌ sherpa-ncnn 不适合追求音质
只适合：

离线设备

嵌入式

低功耗

SDK 分发

✅ 你的最佳组合（强烈建议）
你之前思路其实是对的，我帮你升级成工业版：

⭐ 推荐架构（高质量）
文本
  ↓
高质量断句（LLM / spaCy / 自定义规则）
  ↓
Kokoro 分词（可选）
  ↓
Piper（sherpa-ncnn）推理
👉 或者更狠一点：

短文本 → Piper
长文本 / 重要内容 → Kokoro
六、最终结论（帮你省时间）
如果只在 sherpa-ncnn 里选：

👉 唯一答案：Piper (VITS)
👉 最推荐模型：medium 系列（平衡质量 + 速度）

七、我给你一句人话总结
sherpa-ncnn：
👉 “部署神器”

Piper：
👉 “够用但不惊艳”

Kokoro / QwenTTS：
👉 “效果王者，但重”

如果你要，我可以帮你直接给一套：

👉 “媲美 Kokoro，但用 Piper 跑”的完整工程方案（含断句+韵律优化）
这个才是真正工业级的。

记忆几乎已满
记忆满后，回复的个性化程度将会降低。升级以扩展记忆，或管理现有记忆。

管理

获取更多



"""
sid = 3
print(f"Generating speech (sid={sid}, voice=zf_001)...", flush=True)
start = time.perf_counter()
audio = tts.generate(text, sid=sid, speed=1.0)
elapsed = time.perf_counter() - start
print(f"Sample rate: {audio.sample_rate}", flush=True)
print(f"Audio samples: {len(audio.samples)}", flush=True)
print(f"Duration: {len(audio.samples) / audio.sample_rate:.2f}s", flush=True)
print(f"Generation time: {elapsed:.2f}s", flush=True)
print(f"RTF: {elapsed / (len(audio.samples) / audio.sample_rate):.3f}", flush=True)
output_path = "F:/pythonx/mytts/kokoro-onnx/sherpa-model/test_output.wav"
sherpa_onnx.write_wave(output_path, audio.samples, audio.sample_rate)
print(f"Saved to: {output_path}", flush=True)
print("Done!", flush=True)
