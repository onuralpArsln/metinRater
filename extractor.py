import sys
import os
import glob
from bs4 import BeautifulSoup


def extract_products(target_file='target.html'):
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except FileNotFoundError:
        print(f"Error: '{target_file}' not found.")
        return [], []

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
    print(f"[{target_file}] Found {len(product_spans)} products. Organic: {total_found}. Sponsored: {sponsored_count}")
    
    # Get the first 20 and last 20
    first_20 = product_names[:20]
    last_20 = product_names[-20:] if total_found >= 20 else []

    return first_20, last_20

if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    target_files = []
    if arg:
        if os.path.isdir(arg):
            target_files = glob.glob(os.path.join(arg, "*.html"))
        elif os.path.isfile(arg):
            target_files = [arg]
        else:
            print(f"Error: '{arg}' is not a valid file or directory.")
            sys.exit(1)
    else:
        # Defaults
        if os.path.exists('target.html'):
            target_files = ['target.html']
        elif os.path.isdir('targets'):
            target_files = glob.glob("targets/*.html")
        else:
            print("Error: No target specified and default 'target.html' or 'targets/' not found.")
            sys.exit(1)

    if not target_files:
        print("No HTML files found to process.")
        sys.exit(0)

    all_successful = []
    all_unsuccessful = []

    for target in target_files:
        succ, unsucc = extract_products(target)
        all_successful.extend(succ)
        all_unsuccessful.extend(unsucc)

    # Write aggregated results
    with open('successful.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_successful))
    print(f"Successfully wrote {len(all_successful)} total items to successful.txt")
        
    with open('unsuccessful.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_unsuccessful))
    print(f"Successfully wrote {len(all_unsuccessful)} total items to unsuccessful.txt")
