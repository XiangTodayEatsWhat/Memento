import os
import json
import torch
import tqdm


class Ego4D:
    def __init__(self, embed_dir: str, frame_fps: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dir = embed_dir.rstrip("/")
        self.frame_fps = frame_fps
        self.metadata = self.get_metadata()

    def get_metadata(self):
        metadata_path = f"{self.embed_dir}_metadata.json"
        if os.path.isfile(metadata_path):
            with open(metadata_path) as f:
                return json.load(f)
        metadata = {}
        for name in tqdm.tqdm(os.listdir(self.embed_dir), desc=f"prepare {metadata_path}"):
            if not name.endswith(".pt"):
                continue
            path = os.path.join(self.embed_dir, name)
            key = os.path.splitext(name)[0]
            try:
                emb = torch.load(path, map_location="cpu", weights_only=True)
                if isinstance(emb, dict):
                    emb = next(iter(emb.values()))
                n = emb.shape[0] if hasattr(emb, "shape") else len(emb)
                duration = (n - 1) / self.frame_fps
            except Exception:
                duration = 0.0
            metadata[key] = {"duration": duration, "path": path}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return metadata
