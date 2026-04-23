import onnx  
import os  
print("Loading...", flush=True)  
model = onnx.load(os.path.join("F:/pythonx/mytts/kokoro-onnx/src", "kokoro-v1.1-zh.onnx"))  
print("Loaded", flush=True)  
for k,v in [("sample_rate","24000"),("model_type","kokoro"),("version","1"),("language","zh"),("has_espeak","1"),("n_speakers","1")]:  
    e = model.metadata_props.add()  
    e.key = k  
    e.value = v  
    print(f"  {k}={v}", flush=True)  
print("Saving...", flush=True)  
onnx.save(model, os.path.join("F:/pythonx/mytts/kokoro-onnx/src", "kokoro-v1.1-zh-sherpa.onnx"))  
print("Done!", flush=True)  
