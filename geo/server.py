# pip install fastapi uvicorn python-multipart torch torchvision transformers pillow numpy
import io, copy
import numpy as np, torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPVisionModelWithProjection

torch.set_num_threads(2)

transform = T.Compose([
    T.Resize(224), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]),
])

class MLPHead(nn.Module):
    def __init__(self, n, in_dim, hidden=1024, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, n))
    def forward(self, x):
        return self.net(x)

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 2*6371*np.arcsin(np.sqrt(a))

sd = torch.load("finetuned.pt", map_location="cpu", mmap=True)
clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").eval()
head = MLPHead(sd["head.net.3.weight"].shape[0], sd["head.net.0.weight"].shape[1])

if any(k.startswith("backbone.") for k in sd):
    class Backbone(nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.vision, self.proj = clip.vision_model, clip.visual_projection
        def forward(self, x):
            return self.proj(self.vision(pixel_values=x).pooler_output)
    class FineTuned(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone, self.head = backbone, head
        def forward(self, x):
            return self.head(self.backbone(x))
    net = FineTuned(Backbone(clip), head)
    net.load_state_dict(sd); net.eval()
    run = lambda x: net(x)
else:
    has_proj = any(k.startswith("proj.") for k in sd)
    class LastBlockModel(nn.Module):
        def __init__(self, clip, head, has_proj):
            super().__init__()
            self.block = copy.deepcopy(clip.vision_model.encoder.layers[-1])
            self.ln = copy.deepcopy(clip.vision_model.post_layernorm)
            self.proj = copy.deepcopy(clip.visual_projection) if has_proj else nn.Identity()
            self.head = head
        def forward(self, h):
            out = self.block(h.float(), None, None)
            h = out[0] if isinstance(out, tuple) else out
            return self.head(self.proj(self.ln(h[:, 0])))
    net = LastBlockModel(clip, head, has_proj)
    net.load_state_dict(sd); net.eval()
    run = lambda x: net(clip.vision_model(pixel_values=x, output_hidden_states=True).hidden_states[-2])
del sd

c = np.load("centres.npz")
cell_lat, cell_lon, classes = c["lat"], c["lon"], c["classes"]
assert len(cell_lat) == head.net[3].out_features, "centres.npz is from a different run"
S = torch.tensor(np.nan_to_num(np.exp(-haversine(cell_lat[:, None], cell_lon[:, None],
                                                 cell_lat[None, :], cell_lon[None, :]) / 1492.7)),
                 dtype=torch.float32)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(run(x), dim=1)
        best = (p @ S).argmax(1).item()
        top_p, top_i = p[0].topk(3)
    return {
        "lat": float(cell_lat[best]), "lon": float(cell_lon[best]), "cell": str(classes[best]),
        "top3": [{"cell": str(classes[i]), "prob": float(q), "lat": float(cell_lat[i]), "lon": float(cell_lon[i])}
                 for q, i in zip(top_p, top_i)],
    }