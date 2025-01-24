import bittensor as bt
import clip
import torch
from PIL import Image

from webgenie.constants import HTML_EXTENSION, IMAGE_EXTENSION
from webgenie.rewards.visual_reward.common.inpaint_image import inpaint_image


def rescale(image_path):
    # Load the image
    with Image.open(image_path) as img:
        width, height = img.size

        # Determine which side is shorter
        if width < height:
            # Width is shorter, scale height to match the width
            new_size = (width, width)
        else:
            # Height is shorter, scale width to match the height
            new_size = (height, height)

        # Resize the image while maintaining aspect ratio
        img_resized = img.resize(new_size, Image.LANCZOS)

        return img_resized


def calculate_clip_similarity(image_path1, image_path2, model, preprocess, device):
    # Load and preprocess images
    image1 = preprocess(rescale(image_path1)).unsqueeze(0).to(device)
    image2 = preprocess(rescale(image_path2)).unsqueeze(0).to(device)

    # Calculate features
    with torch.no_grad():
        image_features1 = model.encode_image(image1)
        image_features2 = model.encode_image(image2)

    # Normalize features
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    image_features2 /= image_features2.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity
    similarity = (image_features1 @ image_features2.T).item()

    return similarity


def calculate_embedding_vector(image_path, model, preprocess, device):
    image = preprocess(rescale(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features
    

async def calculate_clip_score(predict_html_path_list, original_html_path):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    original_img_path = original_html_path.replace(HTML_EXTENSION, f"_inpainted{IMAGE_EXTENSION}")
    await inpaint_image(original_html_path, original_img_path)
    original_embedding_vector = calculate_embedding_vector(original_img_path, model, preprocess, device)
    
    results = []
    for predict_html_path in predict_html_path_list:
        try:
            predict_img_path = predict_html_path.replace(HTML_EXTENSION, f"_inpainted{IMAGE_EXTENSION}")
            await inpaint_image(predict_html_path, predict_img_path)
            predict_embedding_vector = calculate_embedding_vector(predict_img_path, model, preprocess, device)

            score = (original_embedding_vector @ predict_embedding_vector.T).item()
            results.append(score)
        except Exception as e:
            bt.logging.error(f"Error calculating clip score for {predict_html_path}: {e}")
            results.append(0)

    return results