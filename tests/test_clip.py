from init_test import init_test

init_test()

from metrics.text_mask import get_text_contour_image

mask = get_text_contour_image("tests/data/test.png")
mask.save("tests/data/mask.png")
