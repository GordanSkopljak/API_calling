from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List
import base64
import json
import os
import sys
import time

import pandas as pd
from PIL import Image
from datasets import load_dataset
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

MODEL_NAME = "gpt-4o"
DATASET_NAME = "ashraq/fashion-product-images-small"
DATASET_SLICE = "train[:5]"
DEFAULT_PRICE = 49.99
SLEEP_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 60
PRODUCT_IMAGES_DIR = Path("product_images")
OUTPUT_LISTINGS_DIR = Path("output_listings")
OUTPUT_FILENAME = "generated_listings.json"


@dataclass
class StepResult:
    data: Any = None
    error: str | None = None


@dataclass
class PipelineResult:
    listings: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    output_path: Path | None = None
    saved: bool = False
    verified: bool = False
    verified_count: int = 0
    success: bool = False


class ProductListingSchema(BaseModel):
    title: str
    description: str
    features: List[str]
    keywords: str


def format_error_message(
    function_name: str,
    error_type: str,
    context: str,
    suggestion: str,
    details: str | None = None,
) -> str:
    message = f"{function_name} | {error_type} | {context} | Suggestion: {suggestion}"
    if details:
        message = f"{message} | Details: {details}"
    return message


def initialize_client() -> OpenAI:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        sys.exit(1)
    return OpenAI(timeout=REQUEST_TIMEOUT_SECONDS)


def initialize_workspace() -> tuple[Path, Path]:
    PRODUCT_IMAGES_DIR.mkdir(exist_ok=True)
    OUTPUT_LISTINGS_DIR.mkdir(exist_ok=True)
    return PRODUCT_IMAGES_DIR, OUTPUT_LISTINGS_DIR


def build_fallback_products(images_dir: Path) -> StepResult:
    products_data = [
        {
            "id": 1,
            "name": "Classic Leather Sneakers",
            "price": 89.99,
            "category": "Footwear",
            "image_path": str(images_dir / "sneakers.jpg"),
        },
        {
            "id": 2,
            "name": "Minimalist Smart Watch",
            "price": 149.50,
            "category": "Electronics",
            "image_path": str(images_dir / "watch.jpg"),
        },
        {
            "id": 3,
            "name": "Ergonomic Office Chair",
            "price": 249.99,
            "category": "Furniture",
            "image_path": str(images_dir / "chair.jpg"),
        },
    ]

    try:
        for product in products_data:
            image_path = Path(product["image_path"])
            if not image_path.exists():
                placeholder = Image.new("RGB", (300, 300), color=(73, 109, 137))
                placeholder.save(image_path)

        return StepResult(data=pd.DataFrame(products_data))
    except Exception:
        return StepResult(error="failed to build fallback products")


def load_product_data(images_dir: Path) -> StepResult:
    print("--- Preparing E-Commerce Dataset ---")
    try:
        print("Loading HuggingFace Fashion Product dataset...")
        dataset = load_dataset(DATASET_NAME, split=DATASET_SLICE)
        products_df = pd.DataFrame(dataset)

        processed_products = []
        for _, row in products_df.iterrows():
            img_path = images_dir / f"product_{row['id']}.jpg"
            if isinstance(row["image"], Image.Image):
                row["image"].save(img_path)

            processed_products.append(
                {
                    "id": row["id"],
                    "name": row.get("productDisplayName", f"Product {row['id']}"),
                    "price": DEFAULT_PRICE,
                    "category": row.get("masterCategory", "Fashion"),
                    "image_path": str(img_path),
                }
            )

        return StepResult(data=pd.DataFrame(processed_products))
    except Exception as exc:
        print(
            f"WARNING: [load_product_data] {type(exc).__name__}: "
            f"Could not fetch via HuggingFace framework: {exc}"
        )
        print("Defaulting to custom local data infrastructure pipeline...")
        fallback_result = build_fallback_products(images_dir)
        if fallback_result.error:
            return StepResult(
                error=format_error_message(
                    "load_product_data",
                    "FallbackBuildError",
                    f"images_dir={images_dir}",
                    "Check that the fallback image directory is writable.",
                    fallback_result.error,
                )
            )
        return StepResult(data=fallback_result.data)


def encode_image_to_base64(image_path: Path, product_id: Any, product_name: str) -> StepResult:
    context = f"product_id={product_id}, product_name={product_name}, image_path={image_path}"
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        return StepResult(data=encoded)
    except FileNotFoundError:
        return StepResult(
            error=format_error_message(
                "encode_image_to_base64",
                "FileNotFoundError",
                context,
                "Verify the product image exists and the path matches the dataset entry.",
            )
        )
    except PermissionError as exc:
        return StepResult(
            error=format_error_message(
                "encode_image_to_base64",
                "PermissionError",
                context,
                "Check file permissions or close any application locking the image.",
                str(exc),
            )
        )


def create_product_listing_prompt(
    product_name: str,
    price: float,
    category: str,
    additional_info: str | None = None,
) -> str:
    return f"""You are an expert e-commerce copywriter. Analyze the product image and create a compelling product listing.

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


def generate_listing_from_api(
    prompt: str,
    encoded_image: str,
    client: OpenAI,
    product_id: Any,
    product_name: str,
) -> StepResult:
    """Send the prompt and image to the Responses API and return structured output."""
    context = f"product_id={product_id}, product_name={product_name}"
    try:
        response = client.responses.parse(
            model=MODEL_NAME,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    ],
                }
            ],
            text_format=ProductListingSchema,
        )

        parsed = response.output_parsed
        return StepResult(
            data={
                "title": parsed.title,
                "description": parsed.description,
                "features": parsed.features,
                "keywords": parsed.keywords,
            }
        )
    except ValidationError as exc:
        return StepResult(
            error=format_error_message(
                "generate_listing_from_api",
                "ValidationError",
                context,
                "Inspect the model response shape and update the schema or prompt.",
                str(exc),
            )
        )
    except AuthenticationError as exc:
        return StepResult(
            error=format_error_message(
                "generate_listing_from_api",
                "AuthenticationError",
                context,
                "Check OPENAI_API_KEY in your environment and confirm it is valid.",
                str(exc),
            )
        )
    except RateLimitError as exc:
        return StepResult(
            error=format_error_message(
                "generate_listing_from_api",
                "RateLimitError",
                context,
                "Slow the batch rate or retry after the rate limit resets.",
                str(exc),
            )
        )
    except APITimeoutError as exc:
        return StepResult(
            error=format_error_message(
                "generate_listing_from_api",
                "APITimeoutError",
                context,
                "Increase the timeout or retry with smaller batches.",
                str(exc),
            )
        )
    except APIConnectionError as exc:
        return StepResult(
            error=format_error_message(
                "generate_listing_from_api",
                "APIConnectionError",
                context,
                "Check network access, DNS, and firewall or proxy settings.",
                str(exc),
            )
        )
    except APIStatusError as exc:
        return StepResult(
            error=format_error_message(
                "generate_listing_from_api",
                type(exc).__name__,
                context,
                "Inspect the HTTP status and response body from OpenAI.",
                f"status_code={getattr(exc, 'status_code', 'unknown')}; {exc}",
            )
        )
    except APIError as exc:
        return StepResult(
            error=format_error_message(
                "generate_listing_from_api",
                "APIError",
                context,
                "Review the OpenAI API error details and retry if needed.",
                str(exc),
            )
        )
    except Exception as exc:
        return StepResult(
            error=format_error_message(
                "generate_listing_from_api",
                "UnexpectedError",
                context,
                "Inspect the traceback and confirm the request payload is valid.",
                str(exc),
            )
        )


def build_listing_payload(row: pd.Series, generated_listing: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": {
            "name": row["name"],
            "price": row["price"],
            "category": row["category"],
        },
        "generated_listing": generated_listing,
    }


def save_generated_listings(output_path: Path, final_listings: dict[str, Any]) -> StepResult:
    try:
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(final_listings, output_file, indent=4, ensure_ascii=False)
        return StepResult(data=True)
    except Exception as exc:
        return StepResult(
            error=format_error_message(
                "save_generated_listings",
                type(exc).__name__,
                f"output_path={output_path}",
                "Verify the destination folder exists and is writable.",
                str(exc),
            )
        )


def verify_generated_listings(output_path: Path) -> StepResult:
    try:
        with open(output_path, "r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
        if not isinstance(payload, dict):
            return StepResult(
                error=format_error_message(
                    "verify_generated_listings",
                    "ValueError",
                    f"output_path={output_path}",
                    "Ensure the JSON file contains a product-id mapping.",
                )
            )
        return StepResult(data={"payload": payload, "record_count": len(payload)})
    except FileNotFoundError as exc:
        return StepResult(
            error=format_error_message(
                "verify_generated_listings",
                "FileNotFoundError",
                f"output_path={output_path}",
                "Confirm the save step completed and the file path is correct.",
                str(exc),
            )
        )
    except json.JSONDecodeError as exc:
        return StepResult(
            error=format_error_message(
                "verify_generated_listings",
                "JSONDecodeError",
                f"output_path={output_path}",
                "Open the JSON file and fix the malformed content before verifying again.",
                f"line={exc.lineno}, column={exc.colno}, msg={exc.msg}",
            )
        )
    except Exception as exc:
        return StepResult(
            error=format_error_message(
                "verify_generated_listings",
                type(exc).__name__,
                f"output_path={output_path}",
                "Inspect the output file and rerun verification.",
                str(exc),
            )
        )


def run_automation_pipeline(df: pd.DataFrame, client: OpenAI, output_dir: Path) -> PipelineResult:
    pipeline_result = PipelineResult()
    print("--- Starting Generation Pipeline ---")

    for _, row in df.iterrows():
        product_id = row["id"]
        product_name = row["name"]
        print(f"Processing Product ID {product_id}: '{product_name}'...")

        image_result = encode_image_to_base64(Path(row["image_path"]), product_id, product_name)
        if image_result.error:
            print(f"ERROR: {image_result.error}")
            pipeline_result.failures.append(
                {"product_id": product_id, "reason": image_result.error}
            )
            continue

        prompt = create_product_listing_prompt(product_name, row["price"], row["category"])
        listing_result = generate_listing_from_api(
            prompt,
            image_result.data,
            client,
            product_id,
            product_name,
        )

        if listing_result.error:
            print(f"ERROR: {listing_result.error}")
            pipeline_result.failures.append(
                {"product_id": product_id, "reason": listing_result.error}
            )
        else:
            print(f"Listing created successfully for Product {product_id}.")
            pipeline_result.listings[str(product_id)] = build_listing_payload(row, listing_result.data)
            time.sleep(SLEEP_SECONDS)

    output_path = output_dir / OUTPUT_FILENAME
    pipeline_result.output_path = output_path

    save_result = save_generated_listings(output_path, pipeline_result.listings)
    if save_result.error:
        print(f"ERROR: {save_result.error}")
        pipeline_result.failures.append(
            {"product_id": "output", "reason": save_result.error}
        )
        return pipeline_result

    pipeline_result.saved = True

    verify_result = verify_generated_listings(output_path)
    if verify_result.error:
        print(f"ERROR: {verify_result.error}")
        pipeline_result.failures.append(
            {"product_id": "output", "reason": verify_result.error}
        )
        return pipeline_result

    pipeline_result.verified = True
    pipeline_result.verified_count = verify_result.data["record_count"]
    if pipeline_result.verified_count == 0:
        print("ERROR: No listings were generated; refusing to report success.")
        pipeline_result.failures.append(
            {"product_id": "output", "reason": "no listings generated"}
        )
        return pipeline_result

    if pipeline_result.verified_count != len(pipeline_result.listings):
        print("ERROR: Verified output count does not match in-memory listings.")
        pipeline_result.failures.append(
            {"product_id": "output", "reason": "verified count mismatch"}
        )
        return pipeline_result

    pipeline_result.success = True
    print(f"\nPipeline Process Complete! Stored at: '{output_path}'")
    return pipeline_result


def main() -> None:
    client = initialize_client()
    images_dir, output_dir = initialize_workspace()
    products_result = load_product_data(images_dir)

    if products_result.data is None:
        print("ERROR: Product data could not be prepared.")
        sys.exit(1)

    print(f"Target setup established. Total items to process: {len(products_result.data)}\n")
    pipeline_result = run_automation_pipeline(products_result.data, client, output_dir)
    print(
        f"Completed with {len(pipeline_result.listings)} successes and "
        f"{len(pipeline_result.failures)} failures."
    )
    if not pipeline_result.success:
        sys.exit(1)


if __name__ == "__main__":
    main()
