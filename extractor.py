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
    
    # Extract text and strip whitespace to clean it up
    product_names = [span.get_text(strip=True) for span in product_spans if span.get_text(strip=True)]
    
    total_found = len(product_names)
    print(f"Found a total of {total_found} product names in 'target.html'")
    
    if total_found < 40:
        print(f"Warning: You requested the first 20 and last 20, but only {total_found} were found. Some items may overlap.")
    
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
