import uuid

from bs4 import BeautifulSoup

from webgenie.constants import DEFAULT_LOAD_TIME, HTML_EXTENSION
from webgenie.rewards.visual_reward.common.take_screenshot import take_screenshot


def erase_texts(input_file_path, output_file_path):
    # Read the input HTML file
    with open(input_file_path, 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')

    def update_style(element, property_name, value):
        # Update the element's style attribute with the given property and value
        # Adding !important to ensure the style overrides others
        important_value = f"{value} !important"
        styles = element.attrs.get('style', '').split(';')
        updated_styles = [s for s in styles if not s.strip().startswith(property_name) and len(s.strip()) > 0]
        updated_styles.append(f"{property_name}: {important_value}")
        element['style'] = '; '.join(updated_styles).strip()

    # Assign a unique color to text within each text-containing element
    text_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'span', 'a', 'b', 'li', 'table', 'td', 'th', 'button', 'footer', 'header', 'figcaption', 'label']  # Add more tags as needed
    for tag in soup.find_all(text_tags):
        update_style(tag, 'color', 'transparent')
        
    # Write the modified HTML to a new file
    with open(output_file_path, 'w') as file:
        file.write(str(soup))


async def inpaint_image(url, output_file_path, load_time = DEFAULT_LOAD_TIME):
    erased_html_path = f'{url.replace(HTML_EXTENSION, "_erased.html")}'
    erase_texts(url, erased_html_path)
    await take_screenshot(erased_html_path, output_file_path, load_time)

