from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI()

import os
import time
import base64
import json
import pandas as pd
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel
from typing import List

# ==========================================
# STEP 1: INITIALIZE OPENAI CLIENT
# ==========================================
# Instantiating the client exactly as your setup block demands
client = OpenAI()

# ==========================================
# STRUCTURAL VALIDATION SCHEMA (PYDANTIC)
# ==========================================
class ProductListingSchema(BaseModel):
    title: str
    description: str
    features: List[str]
    keywords: str

# ==========================================
# STEP 2: PREPARE DATASET AND WORKSPACE
# ==========================================
images_dir = Path("product_images")
images_dir.mkdir(exist_ok=True)
output_dir = Path("output_listings")
output_dir.mkdir(exist_ok=True)

print("--- Preparing E-Commerce Dataset ---")
try:
    print("Loading HuggingFace Fashion Product dataset...")
    dataset = load_dataset("ashraq/fashion-product-images-small", split="train[:5]")
    products_df = pd.DataFrame(dataset)

    processed_products = []
    for idx, row in products_df.iterrows():
        img_path = images_dir / f"product_{row['id']}.jpg"
        if isinstance(row['image'], Image.Image):
            row['image'].save(img_path)

        processed_products.append({
            "id": row['id'],
            "name": row.get('productDisplayName', f"Product {row['id']}"),
            "price": 49.99,
            "category": row.get('masterCategory', 'Fashion'),
            "image_path": str(img_path)
        })
    products_df = pd.DataFrame(processed_products)

except Exception as e:
    print(f"⚠ Could not fetch via HuggingFace framework: {e}")
    print("Defaulting to custom local data infrastructure pipeline...")

    products_data = [
        {"id": 1, "name": "Classic Leather Sneakers", "price": 89.99, "category": "Footwear", "image_path": "product_images/sneakers.jpg"},
        {"id": 2, "name": "Minimalist Smart Watch", "price": 149.50, "category": "Electronics", "image_path": "product_images/watch.jpg"},
        {"id": 3, "name": "Ergonomic Office Chair", "price": 249.99, "category": "Furniture", "image_path": "product_images/chair.jpg"}
    ]
    products_df = pd.DataFrame(products_data)

    for p in products_data:
        p_path = Path(p["image_path"])
        if not p_path.exists():
            img = Image.new('RGB', (300, 300), color=(73, 109, 137))
            img.save(p_path)

print(f"✓ Target setup established. Total items to process: {len(products_df)}\n")

# ==========================================
# STEP 3: ENCODE IMAGES TO BASE64
# ==========================================
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"❌ Error: Image file not located at path: {image_path}")
        return None

# ==========================================
# STEP 4: PROMPT GENERATION ENGINE
# ==========================================
def create_product_listing_prompt(product_name, price, category, additional_info=None):
    prompt = f"""You are an expert e-commerce copywriter. Analyze the product image and create a compelling product listing.

Product Information:
- Name: {product_name}
- Price: ${price:.2f}
- Category: {category}
{f'- Additional Info: {additional_info}' if additional_info else ''}

Please examine the visual content of the image closely. Describe only details you can actually see: color, pattern, material, fit, logos, and design elements. Do not invent or assume brand affiliations, licensing, or details not visible in the image.

Create a professional product listing that includes:
1. Product Title (SEO-friendly, 60 characters max)
2. Product Description (detailed, grounded in what's visible in the image, 150-200 words)
3. Key Features (bullet points, 5-7 items)
4. SEO Keywords (comma-separated, 10-15 keywords)"""
    return prompt

# ==========================================
# STEP 5: USING CLIENT.RESPONSES.PARSE
# ==========================================
def generate_listing_from_api(prompt, encoded_image):
    """Sends the prompt + image using the Responses API's structured-output parser."""
    try:
        response = client.responses.parse(
            model="gpt-4o",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded_image}"}
                ]
            }],
            text_format=ProductListingSchema
        )

        parsed = response.output_parsed

        return {
            "title": parsed.title,
            "description": parsed.description,
            "features": parsed.features,
            "keywords": parsed.keywords
        }

    except Exception as e:
        print(f"❌ Critical API Exception: {e}")
        return None

# ==========================================
# STEP 6: BATCH PROCESSING SYSTEM
# ==========================================
def run_automation_pipeline(df):
    final_listings = {}
    print("--- Starting Generation Pipeline ---")

    for idx, row in df.iterrows():
        p_id = row['id']
        p_name = row['name']
        print(f"Processing Product ID {p_id}: '{p_name}'...")

        base64_str = encode_image_to_base64(row['image_path'])
        if not base64_str:
            continue

        prompt = create_product_listing_prompt(p_name, row['price'], row['category'])
        result = generate_listing_from_api(prompt, base64_str)

        if result:
            print(f"✓ Listing created successfully for Product {p_id}.")
            final_listings[str(p_id)] = {
                "metadata": {"name": p_name, "price": row['price'], "category": row['category']},
                "generated_listing": result
            }
        else:
            print(f"❌ Failed processing sequence for product ID: {p_id}")

        time.sleep(2)

    output_path = output_dir / "generated_listings.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_listings, f, indent=4, ensure_ascii=False)

    print(f"\n🏆 Pipeline Process Complete! Stored at: '{output_path}'")
    return final_listings

if __name__ == "__main__":
    results = run_automation_pipeline(products_df)