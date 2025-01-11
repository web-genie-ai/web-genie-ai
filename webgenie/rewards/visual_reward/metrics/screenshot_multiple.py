import os
from playwright.sync_api import sync_playwright
from PIL import Image
from tqdm import tqdm

def take_screenshots(html_files, output_files, page_load_time = 1000, do_it_again=False):
    with sync_playwright() as p:
        browser = p.chromium.launch()  # You can also use 'firefox' or 'webkit'
        for html_file, output_file in tqdm(zip(html_files, output_files), desc="Screenshoting HTML files"):

            if os.path.exists(html_file):
                html_file = "file://" + os.path.abspath(html_file)

            if os.path.exists(output_file) and not do_it_again:
                print(f"{output_file} exists!")
                continue

            try:
                page = browser.new_page()
                page.goto(html_file, timeout=60000)
                page.wait_for_timeout(page_load_time)
                page.screenshot(path=output_file, full_page=True, animations="disabled", timeout=60000)
                page.close()
            except Exception as e:
                print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
                # Generate a blank image 
                img = Image.new('RGB', (1280, 960), color = 'white')
                img.save(output_file)        
        browser.close()
