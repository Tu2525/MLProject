import torch
import torch.nn as nn
import torchvision.models as models
from transformers import (
    GPT2LMHeadModel, 
    VisionEncoderDecoderModel, 
    ViTImageProcessor, 
    AutoTokenizer,
    BlipProcessor, 
    BlipForConditionalGeneration
)

# -----------------------------------------------------------------------------
# 1. Custom ResNet + GPT-2 (Training from Scratch)
# -----------------------------------------------------------------------------
class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.projection = nn.Linear(2048, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.projection(features)
        features = self.bn(features)
        return features

class ResNetGPT2(nn.Module):
    def __init__(self, max_seq_len=40):
        super(ResNetGPT2, self).__init__()
        self.encoder = ResNetEncoder(embed_dim=768)
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.max_seq_len = max_seq_len

    def forward(self, images, input_ids, attention_mask):
        image_embeds = self.encoder(images)
        token_embeds = self.gpt2.transformer.wte(input_ids)
        inputs_embeds = torch.cat((image_embeds.unsqueeze(1), token_embeds), dim=1)
        batch_size = images.shape[0]
        ones = torch.ones(batch_size, 1).to(images.device)
        attention_mask = torch.cat((ones, attention_mask), dim=1)
        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.logits

    def generate_caption(self, image, tokenizer, max_length=20, temperature=1.0):
        self.eval()
        with torch.no_grad():
            image_embed = self.encoder(image.unsqueeze(0))
            inputs_embeds = image_embed.unsqueeze(1)
            generated_tokens = []
            for _ in range(max_length):
                outputs = self.gpt2(inputs_embeds=inputs_embeds)
                logits = outputs.logits[:, -1, :] / temperature
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                if next_token.item() == tokenizer.eos_token_id:
                    break
                generated_tokens.append(next_token.item())
                next_token_embed = self.gpt2.transformer.wte(next_token)
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
            return tokenizer.decode(generated_tokens, skip_special_tokens=True)

# -----------------------------------------------------------------------------
# 2. ViT + GPT-2 (Pre-trained SOTA 1)
# -----------------------------------------------------------------------------
class ViTGPT2Captioner(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading ViT-GPT2 model...")
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    def generate_caption(self, image, **kwargs):
        self.eval()
        with torch.no_grad():
            pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.model.device)
            output_ids = self.model.generate(pixel_values, max_length=20, num_beams=4)
            preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            return preds[0].strip()

# -----------------------------------------------------------------------------
# 3. BLIP (Pre-trained SOTA 2 - Best)
# -----------------------------------------------------------------------------
class BLIPCaptioner(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading BLIP model (Salesforce/blip-image-captioning-large)...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    def generate_caption(self, image, **kwargs):
        self.eval()
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
            output_ids = self.model.generate(**inputs, max_length=50, num_beams=5, repetition_penalty=1.2, min_length=5)
            caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
            return caption

# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def get_model(config):
    if config.MODEL_TYPE == "resnet_gpt2":
        return ResNetGPT2()
    elif config.MODEL_TYPE == "vit_gpt2":
        return ViTGPT2Captioner()
    elif config.MODEL_TYPE == "blip":
        return BLIPCaptioner()
    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")

