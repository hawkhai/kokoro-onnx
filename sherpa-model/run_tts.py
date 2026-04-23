import time  
import sherpa_onnx  
BASE = "F:/pythonx/mytts/kokoro-onnx"  
MODEL_DIR = BASE + "/sherpa-model"  
MODEL_ONNX = BASE + "/src/kokoro-v1.1-zh-sherpa.onnx"  
VOICES_BIN = BASE + "/src/voices-v1.1-zh.bin"  
kokoro_config = sherpa_onnx.OfflineTtsKokoroModelConfig(  
    model=MODEL_ONNX,  
    voices=VOICES_BIN,  
    tokens=MODEL_DIR + "/tokens.txt",  
    lexicon=MODEL_DIR + "/lexicon-zh.txt",  
    data_dir=MODEL_DIR + "/espeak-ng-data",  
    length_scale=1.0,  
)  
model_config = sherpa_onnx.OfflineTtsModelConfig(  
    kokoro=kokoro_config,  
    num_threads=4,  
    debug=True,  
    provider="cpu",  
)  
rule_fsts = ",".join([  
    MODEL_DIR + "/date-zh.fst",  
    MODEL_DIR + "/number-zh.fst",  
    MODEL_DIR + "/phone-zh.fst",  
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
text = "\u4f60\u597d\uff0c\u6b22\u8fce\u4f7f\u7528\u8bed\u97f3\u5408\u6210\u7cfb\u7edf\u3002\u4eca\u5929\u662f2026\u5e744\u670822\u65e5\uff0c\u5929\u6c14\u6674\u6717\u3002"  
sid = 0  
print(f"Generating speech for: {text}", flush=True)  
start = time.perf_counter()  
audio = tts.generate(text, sid=sid, speed=1.0)  
elapsed = time.perf_counter() - start  
print(f"Sample rate: {audio.sample_rate}", flush=True)  
print(f"Audio samples: {len(audio.samples)}", flush=True)  
print(f"Duration: {len(audio.samples) / audio.sample_rate:.2f}s", flush=True)  
print(f"Generation time: {elapsed:.2f}s", flush=True)  
print(f"RTF: {elapsed / (len(audio.samples) / audio.sample_rate):.3f}", flush=True)  
output_path = MODEL_DIR + "/test_output.wav"  
sherpa_onnx.write_wave(output_path, audio.samples, audio.sample_rate)  
print(f"Saved to: {output_path}", flush=True)  
print("Done!", flush=True) 
