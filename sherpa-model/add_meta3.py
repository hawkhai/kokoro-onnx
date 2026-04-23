import onnx
import zipfile
print("Reading voices...", flush=True)
z = zipfile.ZipFile("F:/pythonx/mytts/kokoro-onnx/src/voices-v1.1-zh.bin", "r")
names = sorted([n.replace(".npy","") for n in z.namelist()])
z.close()
speaker_names = ",".join(names)
print(f"Found {len(names)} voices", flush=True)
print("Loading model...", flush=True)
model = onnx.load("F:/pythonx/mytts/kokoro-onnx/src/kokoro-v1.1-zh.onnx")
print("Loaded", flush=True)
meta = {"sample_rate":"24000","model_type":"kokoro","version":"1","language":"zh","has_espeak":"1","n_speakers":str(len(names)),"speaker_names":speaker_names,"style_dim":"256"}
for k,v in meta.items():
    e = model.metadata_props.add()
    e.key = k
    e.value = v
    disp = v[:60] + "..." if len(v) > 60 else v
    print(f"  {k}={disp}", flush=True)
print("Saving...", flush=True)
onnx.save(model, "F:/pythonx/mytts/kokoro-onnx/src/kokoro-v1.1-zh-sherpa.onnx")
print("Done!", flush=True)
