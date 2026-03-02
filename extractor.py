from bs4 import BeautifulSoup

def extract_products():
    try:
        with open('target.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
    except FileNotFoundError:
        print("Error: 'target.html' not found in the current directory.")
        return

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all span elements with class 'product-name'
    product_spans = soup.find_all('span', class_='product-name')
    
    # Extract text and strip whitespace to clean it up, skipping sponsored items
    product_names = []
    sponsored_count = 0
    
    for span in product_spans:
        text = span.get_text(strip=True)
        if not text:
            continue
            
        # Try to find the container div for this specific product
        wrapper = span.find_parent('div', class_='p-card-wrppr')
        if not wrapper:
            # Fallback for target.html structure
            curr = span
            while curr and curr.parent and not (curr.parent.name == 'div' and 'search-result-content' in (curr.parent.get('class') or [])):
                curr = curr.parent
            wrapper = curr
            
        is_sponsored = False
        if wrapper:
            # Check for sponsor image or text
            sponsor_img = wrapper.find('img', src=lambda s: s and 'sponsor' in s.lower() if s else False)
            sponsor_text = wrapper.find('span', class_='sponsor-text')
            is_sponsored = bool(sponsor_img) or bool(sponsor_text)
            
        if is_sponsored:
            sponsored_count += 1
        else:
            product_names.append(text)
    
    total_found = len(product_names)
    print(f"Found a total of {len(product_spans)} products.")
    print(f"Skipped {sponsored_count} sponsored products.")
    print(f"Total organic products extracted: {total_found}")
    
    if total_found < 40:
        print(f"Warning: You requested the first 20 and last 20, but only {total_found} organic products were found. Some items may overlap.")
    
    # Get the first 20 and last 20
    first_20 = product_names[:20]
    last_20 = product_names[-20:] if total_found >= 20 else []

    # Write the first 20 to successful.txt
    with open('successful.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(first_20))
    print(f"Successfully wrote {len(first_20)} items to successful.txt")
        
    # Write the last 20 to unsuccessful.txt
    with open('unsuccessful.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(last_20))
    print(f"Successfully wrote {len(last_20)} items to unsuccessful.txt")

if __name__ == '__main__':
    extract_products()
