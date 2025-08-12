import glob
import json
import os
import re
import shutil
from typing import Any, Dict, List

from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf


# Utility: Make a safe filename from document and element ids
def safe_filename(base: str, ext: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_\-]", "_", base)
    return f"{base}.{ext}"


def extract_and_chunk_md(md_path: str, chunking_strategy: str = "page") -> List[Dict[str, Any]]:
    doc_id = os.path.splitext(os.path.basename(md_path))[0]
    print(f"Processing: {md_path}")
    print(f"doc_id: {doc_id}")
    elements = partition_md(md_path)
    chunks: Dict[str, Any] = {}
    for i, elem in enumerate(elements):
        page_num = i + 1
        chunk_key = f"{doc_id}_page_{page_num}"
        if chunk_key not in chunks:
            chunks[chunk_key] = {
                "doc_id": doc_id,
                "page_number": page_num,
                "chunk_id": chunk_key,
                "text": [],
                "element_types": [],
            }
        chunks[chunk_key]["text"].append(f"{elem.category}: {elem.text}")
        chunks[chunk_key]["element_types"].append(elem.category)
    output_chunks = list(chunks.values())
    for chunk in chunks.values():
        chunk["text"] = "\n".join(chunk["text"]).strip()
    return output_chunks


def extract_and_chunk_mds(
    md_dir: str, output_dir: str, chunking_strategy: str = "page", write_to_json: bool = True
) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    md_files = glob.glob(os.path.join(md_dir, "*.md"))
    all_chunks = []
    for md_path in md_files:
        chunks = extract_and_chunk_md(md_path, chunking_strategy)
        all_chunks.extend(chunks)
    if write_to_json:
        json_path = os.path.join(output_dir, "all_chunks.jsonl")
        write_mode = "a" if os.path.exists(json_path) else "w"
        with open(json_path, write_mode, encoding="utf-8") as f:
            for chunk in all_chunks:
                json.dump(chunk, f, ensure_ascii=False)
                f.write("\n")
    return all_chunks


def extract_and_chunk_pdf(
    pdf_path: str, output_dir: str, chunking_strategy: str = "page"
) -> List[Dict[str, Any]]:
    # Create images directory if it doesn't exist
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    # Create block images directory if it doesn't exist
    os.makedirs(f"{output_dir}/block_images", exist_ok=True)
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"Processing: {pdf_path}")
    print(f"doc_id: {doc_id}")
    try:
        elements = partition_pdf(
            pdf_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=f"{output_dir}/block_images",
        )
    except Exception as e:
        print(f"Error parsing {pdf_path}: {e}")
        return [{}]

    # Chunk grouping: by page number
    chunks: Dict[str, Any] = {}
    for i, elem in enumerate(elements):
        page_num = getattr(elem.metadata, "page_number", 0)
        print(f"page_num: {page_num}")
        chunk_key = f"{doc_id}_page_{page_num}"
        if chunk_key not in chunks:
            chunks[chunk_key] = {
                "doc_id": doc_id,
                "page_number": page_num,
                "chunk_id": chunk_key,
                "text": [],
                "images": [],
                "element_types": [],
            }
        print(f"elem.category: {elem.category}")
        if elem.category in [
            "NarrativeText",
            "Title",
            "Header",
            "Footer",
            "ListItem",
            "UncategorizedText",
        ]:
            if hasattr(elem, "text") and elem.text:
                chunks[chunk_key]["text"].append(f"{elem.category}: {elem.text}")
                chunks[chunk_key]["element_types"].append(elem.category)
        elif elem.category in ["Image", "FigureCaption", "Table"]:
            # Try to extract image in all ways known (robust)
            img_filename = safe_filename(f"{chunk_key}_img_{i}", "png")
            img_path = os.path.join(f"{output_dir}/images", img_filename)
            print(f"img_path: {img_path}")
            success = False
            try:
                if hasattr(elem, "image_data") and elem.image_data:
                    with open(img_path, "wb") as f:
                        f.write(elem.image_data)
                    success = True
                elif hasattr(elem, "binary_data") and elem.binary_data:
                    with open(img_path, "wb") as f:
                        f.write(elem.binary_data)
                    success = True
                elif (
                    hasattr(elem, "metadata")
                    and hasattr(elem.metadata, "image_path")
                    and elem.metadata.image_path
                ):
                    shutil.copy(elem.metadata.image_path, img_path)
                    success = True
                elif hasattr(elem, "save"):
                    elem.save(img_path)
                    success = True
            except Exception as img_exc:
                print(f"Image extraction failed for {img_filename}: {img_exc} ")
            if success:
                chunks[chunk_key]["images"].append(img_path)
                chunks[chunk_key]["element_types"].append("Image")
    output_chunks = list(chunks.values())
    for chunk in chunks.values():
        chunk["text"] = "\n".join(chunk["text"]).strip()
    return output_chunks


# Main: Extract and chunk all PDFs in directory
def extract_and_chunk_pdfs(
    pdf_dir: str,
    output_dir: str,
    chunking_strategy: str = "page",
    write_to_json: bool = True,
) -> List[Dict[str, Any]]:
    """
    - pdf_dir: path to folder with PDFs
    - output_dir: where to write extracted images
    - chunking_strategy: 'page' (default), 'slide', 'section', 'fixed' (customize as needed)
    - write_to_json: if True, write chunks to json file
    Returns: list of dicts (each dict = 1 chunk: text, images, metadata)
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    all_chunks = []

    for pdf_path in pdf_files:
        chunks = extract_and_chunk_pdf(pdf_path, output_dir, chunking_strategy)
        all_chunks.extend(chunks)

    if write_to_json:
        # if json file already exists, append to it
        json_path = os.path.join(output_dir, "all_chunks.jsonl")
        write_mode = "a" if os.path.exists(json_path) else "w"
        with open(json_path, write_mode, encoding="utf-8") as f:
            for chunk in all_chunks:
                json.dump(chunk, f, ensure_ascii=False)
                f.write("\n")
    print(f"Extracted {len(all_chunks)} chunks from {len(pdf_files)} PDFs.")
    return all_chunks


if __name__ == "__main__":
    # PDF_DIR = (
    #     r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset\repository\PDFs"
    #     r"\biotech_pdfs\experiment"
    # )
    # OUTPUT_DIR = (
    #     r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset\repository\PDFs"
    #     r"\biotech_pdfs\experiment\pdf_extracted_images"
    # )
    # chunks = extract_and_chunk_pdfs(PDF_DIR, OUTPUT_DIR, chunking_strategy="page")

    # json_path = os.path.join(OUTPUT_DIR, "all_chunks.jsonl")
    # with open(json_path, "w", encoding="utf-8") as f:
    #     for chunk in chunks:
    #         json.dump(chunk, f, ensure_ascii=False)
    #         f.write("\n")

    # print(f"Chunk metadata saved to {json_path}")

    # pdf_path = (
    #     r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset\repository\PDFs"
    #     r"\biotech_pdfs\experiment\2502.15646v1.pdf"
    # )
    # pdf_repo_dir = (
    #     r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset\repository\PDFs"
    #     r"\biotech_pdfs\experiment"
    # )
    # output_dir = os.path.join(pdf_repo_dir, "pdf_extracted_images")
    # chunks = extract_and_chunk_pdfs(pdf_repo_dir, output_dir, chunking_strategy="page")
    # print("--------------------------------")
    # print("Chunks:")
    # print(chunks)
    # print("--------------------------------")

    md_dir = r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset\repository\company"
    output_dir = (
        r"D:\projects\machine_learning_workspace\ml-boilerplate\dataset\repository\company\md_extractions"
    )
    chunks = extract_and_chunk_mds(md_dir, output_dir, chunking_strategy="page")
    print("--------------------------------")
    print("Chunks:")
    print(chunks)
    print("--------------------------------")
