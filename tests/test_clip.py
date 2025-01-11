import clip
import torch
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Load and preprocess images
image1 = preprocess(Image.open('path_to_text_image1.jpg')).unsqueeze(0).to(device)
image2 = preprocess(Image.open('path_to_text_image2.jpg')).unsqueeze(0).to(device)

# Compute features and calculate cosine similarity
with torch.no_grad():
    image_features1 = model.encode_image(image1)
    image_features2 = model.encode_image(image2)
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    image_features2 /= image_features2.norm(dim=-1, keepdim=True)
    cosine_similarity = (image_features1 @ image_features2.T).cpu().numpy()

print(f"Cosine similarity: {cosine_similarity.item()}")
