
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import numpy as np

def delta_e_cie2000(lab1, lab2):
    # Extract components of Lab1 and Lab2
    L1, a1, b1 = lab1.lab_l, lab1.lab_a, lab1.lab_b
    L2, a2, b2 = lab2.lab_l, lab2.lab_a, lab2.lab_b
    
    # 1. Calculate differences in lightness (L*)
    delta_L = L1 - L2
    
    # 2. Calculate chroma for both colors
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    
    # 3. Calculate differences in chroma (C*)
    delta_C = C1 - C2
    
    # 4. Calculate hue for both colors
    h1 = np.arctan2(b1, a1)  # Hue of color 1
    h2 = np.arctan2(b2, a2)  # Hue of color 2
    
    # Ensure the hue difference is calculated in the range [0, 360)
    delta_H = h1 - h2
    if delta_H < 0:
        delta_H += 2 * np.pi  # Wrap around 360°
    if delta_H > np.pi:
        delta_H -= 2 * np.pi
    
    # 5. Calculate the Hue difference (ΔH*)
    delta_H_star = 2 * np.sqrt(C1 * C2) * np.sin(delta_H / 2)
    
    # 6. Calculate the rotation term R_T
    delta_theta = h1 - h2 + np.radians(30)
    R_T = -0.17 * np.cos(delta_theta) + 0.24 * np.cos(2 * h1 + np.radians(60)) \
          - 0.32 * np.cos(3 * h1 + np.radians(120)) + 0.2 * np.cos(4 * h1 - np.radians(63))
    
    # 7. Calculate the final Delta E (CIE 2000)
    term1 = delta_L**2
    term2 = delta_C**2
    term3 = delta_H_star**2
    term4 = R_T * delta_C * delta_H_star
    
    delta_E_00 = np.sqrt(term1 + term2 + term3 + term4)
    
    return delta_E_00


def rgb_to_lab(rgb):
    """
    Convert an RGB color to Lab color space.
    RGB values should be in the range [0, 255].
    """
    # Create an sRGBColor object from RGB values
    rgb_color = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True)
    
    # Convert to Lab color space
    lab_color = convert_color(rgb_color, LabColor)
    
    return lab_color



def color_similarity_ciede2000(rgb1, rgb2):
    """
    Calculate the color similarity between two RGB colors using the CIEDE2000 formula.
    Returns a similarity score between 0 and 1, where 1 means identical.
    """
    # Convert RGB colors to Lab
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    
    # Calculate the Delta E (CIEDE2000)
    delta_e = delta_e_cie2000(lab1, lab2)
    
    # Normalize the Delta E value to get a similarity score
    similarity = max(0, 1 - (delta_e / 100))
    return similarity