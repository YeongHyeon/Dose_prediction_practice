import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

class DINOV3FeatureExtractor:

    def __init__(self, model_id=None):
        if model_id is None:
            model_id = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()

        # Disable auto-resize / crop so we control input size
        self.processor.do_resize = False
        self.processor.do_center_crop = False

        self.patch_size = getattr(self.model.config, "patch_size", 16)

    def _pad_to_multiple(self, array, multiple: int):
        """
        Pad a tensor [C,H,W] or [H,W] so H and W become multiples of 'multiple'.
        Pads with zeros on the bottom/right only.
        Returns the padded tensor and padding info.
        """
        if isinstance(array, torch.Tensor):
            if array.ndim == 2:  # [H,W]
                array = array.unsqueeze(0)  # -> [1,H,W]
            elif array.ndim == 3:
                pass  # [C,H,W]
            else:
                raise ValueError(f"Expected 2D or 3D tensor, got shape {array.shape}")

            _, h, w = array.shape
            pad_w = (multiple - (w % multiple)) % multiple
            pad_h = (multiple - (h % multiple)) % multiple

            # pad format for F.pad: (left, right, top, bottom)
            padded = F.pad(array, (0, pad_w, 0, pad_h), value=0)
            return padded

    def extract_features(self, array):
        padded = self._pad_to_multiple(array, self.patch_size)
        inputs = self.processor(images=padded, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        
        num_reg = getattr(self.model.config, "num_register_tokens", 0)
        patch_tokens = last_hidden[1+num_reg:, :]         # [1, N, D]

        # Patch grid size (H', W')
        W, H = padded.shape[-1], padded.shape[-2]
        gh, gw = H // self.patch_size, W // self.patch_size
        patch_map = patch_tokens.reshape(gh, gw, -1)

        return patch_map
    
    def extract_radiomics(self, x, use_minmax=True, use_skew=True, use_kurt=True):
        """
        x: [H, W, C]
        """
        x = torch.tensor(x, dtype=torch.float32)
        feat_mean = torch.mean(x, -1).unsqueeze(-1)
        feat_std = torch.std(x, -1).unsqueeze(-1)
        feats = torch.cat([feat_mean, feat_std], dim=-1)
        names = ["AVG", "SD"]
        
        if use_minmax:
            feat_min = torch.min(x, -1).values.unsqueeze(-1)
            feat_max = torch.max(x, -1).values.unsqueeze(-1)
            feats = torch.cat([feats, feat_min, feat_max], dim=-1)
            names.extend(["MIN", "MAX"])
        if use_skew or use_kurt:
            z = (x - feat_mean) / (feat_std + 1e-12)
            if use_skew:
                feat_skew = (z ** 3).mean(-1).unsqueeze(-1)
                feats = torch.cat([feats, feat_skew], dim=-1)
                names.append("SKEW")
            if use_kurt:
                feat_kurt = (z ** 4).mean(-1).unsqueeze(-1)
                feats = torch.cat([feats, feat_kurt], dim=-1)
                names.append("KURT")

        return feats, names