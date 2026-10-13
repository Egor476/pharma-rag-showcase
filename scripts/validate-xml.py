import sys
from pathlib import Path

from lxml import etree  # type: ignore


def validate_xml_files():
    base_dir = Path(__file__).resolve().parent.parent
    content_dir = base_dir / "content"
    
    if not content_dir.exists():
        print(f"Error: Content directory not found at {content_dir}")
        return False
        
    success = True
    print("Starting XML/DITA structure validation...")
    
    # Gather both .ditamap and .dita files recursively
    files_to_check = list(content_dir.rglob("*.ditamap")) + list(content_dir.rglob("*.dita"))
    
    for xml_file in files_to_check:
        try:
            etree.parse(str(xml_file))
            print(f"  [PASS] {xml_file.relative_to(base_dir)}")
        except (OSError, etree.XMLSyntaxError) as e:
            print(f"  [FAIL] {xml_file.relative_to(base_dir)} parsing failed: {e}")
            success = False
            
    return success

if __name__ == "__main__":
    if not validate_xml_files():
        sys.exit(1)
    else:
        print("All XML/DITA files successfully validated!")
        sys.exit(0)