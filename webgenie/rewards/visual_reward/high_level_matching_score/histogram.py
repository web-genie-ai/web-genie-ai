import numpy as np
from PIL import Image

from webgenie.rewards.visual_reward.common.take_screenshot import take_screenshot
from webgenie.constants import HTML_EXTENSION, IMAGE_EXTENSION

def compute_grayscale_histogram(image_path, bins=256):
    """
    Load the image, convert to grayscale, compute histogram.
    """
    # Load image and convert to grayscale
    img = Image.open(image_path).convert("L")
    np_img = np.array(img)

    # Compute histogram (range 0-255)
    hist, edges = np.histogram(np_img, bins=bins, range=(0, 256))

    # Normalize the histogram so it sums up to 1 (optional)
    hist = hist.astype(float)
    hist /= hist.sum()

    return hist

def compare_histograms(hist1, hist2):
    """
    Compare two 1D histograms (same bin count) using correlation coefficient.
    """
    # Use numpy.corrcoef to get correlation
    corr = np.corrcoef(hist1, hist2)[0, 1]
    return (corr + 1) / 2


async def histogram_matching_score(predict_html_path_list, original_html_path):
    original_img_path = original_html_path.replace(HTML_EXTENSION, IMAGE_EXTENSION)
    await take_screenshot(original_html_path, original_img_path)
    original_hist = compute_grayscale_histogram(original_img_path)
    
    results = []
    for predict_html_path in predict_html_path_list:
        predict_img_path = predict_html_path.replace(HTML_EXTENSION, IMAGE_EXTENSION)
        await take_screenshot(predict_html_path, predict_img_path)
        predict_hist = compute_grayscale_histogram(predict_img_path)
        similarity = compare_histograms(original_hist, predict_hist)
        results.append(similarity)

    return results