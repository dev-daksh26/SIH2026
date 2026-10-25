import os
import sys
import json
import sqlite3
import pymupdf

# Set UTF-8 encoding for terminal output on Windows
sys.stdout.reconfigure(encoding="utf-8")

import main

def test_document_extraction(filename: str, file_path: str):
    print("=" * 70)
    print(f"📄 Testing Document: {filename}")
    print("=" * 70)

    if not os.path.exists(file_path):
        print(f"❌ File not found at: {file_path}")
        return

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    raw_text, ocr_quality = main.run_ocr(file_bytes, filename)
    extracted, conf = main.extract_land_identity(raw_text)

    print(f"Detected Source Language : {extracted['source_language'].upper()}")
    print(f"OCR Quality Score        : {ocr_quality}")
    print(f"Document Type (EN)       : {extracted['document_type']}")
    print(f"Document Type (Native)   : {extracted['document_type_native']}")
    print(f"Serial / Deed No.        : {extracted['serial_number']}")
    print(f"Registration Date        : {extracted['registration_date']}")
    print(f"Mutation Date            : {extracted['mutation_date']}")
    print("-" * 70)
    print("📍 LOCATION:")
    print(f"  District : {extracted['district']}  |  Native: {extracted['district_native']}")
    print(f"  Tehsil   : {extracted['tehsil']}  |  Native: {extracted['tehsil_native']}")
    print(f"  Village  : {extracted['village']}  |  Native: {extracted['village_native']}")
    print("-" * 70)
    print("🔖 IDENTIFIERS:")
    print(f"  Khasra No   : {extracted['khasra_number']}")
    print(f"  Khata No    : {extracted['khata_number']}")
    print(f"  Plot No     : {extracted['plot_number']}")
    print(f"  Property ID : {extracted['property_id']}")
    print("-" * 70)
    print("👤 PERSON & HOLDING:")
    print(f"  Owner (EN)  : {extracted['owner_name']}")
    print(f"  Owner (Nat) : {extracted['owner_name_native']}")
    print(f"  Seller (EN) : {extracted['seller_name']}")
    print(f"  Seller (Nat): {extracted['seller_name_native']}")
    print(f"  Share       : {extracted['share_hissa']}")
    print("-" * 70)
    print("📐 PROPERTY DETAILS:")
    print(f"  Total Area    : {extracted['total_area']} Bigha")
    print(f"  Land Use (EN) : {extracted['land_use']}")
    print(f"  Land Use (Nat): {extracted['land_use_native']}")
    print(f"  Consideration : Rs. {extracted['consideration_amount']:,.2f}" if extracted['consideration_amount'] is not None else "  Consideration : None")
    print(f"  Boundaries EN : {extracted['boundaries_desc']}")
    print(f"  Boundaries Nat: {extracted['boundaries_desc_native']}")
    print("=" * 70)
    print("\n")

def run_all_tests():
    data_dir = os.path.join(os.path.dirname(__file__), "Normal Data")

    test_files = [
        "Hindi1.pdf",
        "hindi2.pdf",
        "hindi3.pdf",
        "1.pdf",
        "Timeline2.pdf"
    ]

    for fname in test_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            test_document_extraction(fname, fpath)
        else:
            print(f"⚠️ Warning: {fname} not found in {data_dir}")

if __name__ == "__main__":
    run_all_tests()
