
PDF_PATH = "output/www_wellsfargo_com/pdfs/brokerage-terms-and-conditions.pdf"
"""
Debug script to test Docling extraction on a single PDF
Run this to see what's happening with Docling
"""
from pathlib import Path

# Configure the path to your PDF
#PDF_PATH = "output/www_wellsfargo_com/pdfs/terms-and-conditions.pdf"

print("="*60)
print("Docling Debug Test")
print("="*60)
print(f"\nTesting PDF: {PDF_PATH}")
print(f"File exists: {Path(PDF_PATH).exists()}")
print(f"File size: {Path(PDF_PATH).stat().st_size:,} bytes")

# Test 1: Simple default converter
print("\n" + "="*60)
print("TEST 1: Simple DocumentConverter with defaults")
print("="*60)
try:
    from docling.document_converter import DocumentConverter
    
    converter = DocumentConverter()
    print("✓ Converter initialized with defaults")
    
    print("\nConverting PDF...")
    result = converter.convert(PDF_PATH)
    print("✓ Conversion completed")
    
    if result and hasattr(result, 'document'):
        doc = result.document
        print(f"✓ Document object retrieved")
        
        # Try markdown export
        try:
            markdown = doc.export_to_markdown()
            print(f"\n✓ Markdown export: {len(markdown)} characters")
            if markdown and len(markdown) > 0:
                print(f"\nFirst 500 characters:")
                print("-" * 60)
                print(markdown[:500])
                print("-" * 60)
            else:
                print("⚠ Markdown is empty")
        except Exception as e:
            print(f"✗ Markdown export failed: {e}")
        
        # Try text export
        try:
            text = doc.export_to_text()
            print(f"\n✓ Text export: {len(text)} characters")
            if text and len(text) > 0:
                print(f"\nFirst 500 characters:")
                print("-" * 60)
                print(text[:500])
                print("-" * 60)
            else:
                print("⚠ Text is empty")
        except Exception as e:
            print(f"✗ Text export failed: {e}")
            
    else:
        print("✗ No document in result")
        
except Exception as e:
    print(f"✗ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: With PdfPipelineOptions
print("\n" + "="*60)
print("TEST 2: DocumentConverter with PdfPipelineOptions")
print("="*60)
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    
    # Try different configurations
    configs_to_test = [
        {"do_ocr": False, "do_table_structure": False},
        {"do_ocr": True, "do_table_structure": False},
        {"do_ocr": False, "do_table_structure": True},
        {"do_ocr": True, "do_table_structure": True},
    ]
    
    for idx, config in enumerate(configs_to_test):
        print(f"\nConfig {idx+1}: {config}")
        try:
            pipeline_options = PdfPipelineOptions(**config)
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: pipeline_options,
                }
            )
            print(f"  ✓ Converter initialized")
            
            result = converter.convert(PDF_PATH)
            
            if result and hasattr(result, 'document'):
                markdown = result.document.export_to_markdown()
                text = result.document.export_to_text()
                
                print(f"  ✓ Markdown: {len(markdown)} chars, Text: {len(text)} chars")
                
                if len(markdown) > 0 or len(text) > 0:
                    print(f"  🎉 SUCCESS! This config works!")
                    print(f"\nSample output:")
                    print("-" * 60)
                    print((markdown or text)[:500])
                    print("-" * 60)
                    break
            else:
                print(f"  ✗ No document returned")
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
except Exception as e:
    print(f"✗ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Debug test complete")
print("="*60)