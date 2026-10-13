import os
from pathlib import Path
import yaml
import lxml.etree as etree

def parse_subject(element):
    """Recursively parse subjectdef elements into a dictionary structure."""
    subject = {}
    key = element.get("keys")
    title = element.get("navtitle")
    
    if key:
        subject["key"] = key
    if title:
        subject["title"] = title
        
    children = []
    for child in element.xpath("./*[local-name()='subjectdef']"):
        child_parsed = parse_subject(child)
        if child_parsed:
            children.append(child_parsed)
            
    if children:
        subject["children"] = children
        
    return subject

def convert_taxonomy(xml_path: Path, yaml_path: Path):
    """Convert DITA subjectScheme XML to YAML taxonomy schema."""
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    
    # Extract root-level metadata
    tax_id = root.get("id", "pharma_product_taxonomy")
    lang = root.get("{http://www.w3.org/XML/1998/namespace}lang", "en-US")
    title_elem = root.find("title")
    title_text = title_elem.text if title_elem is not None else "Taxonomy"
    
    subjects = []
    # Find top-level subjectdefs (children of root that are subjectdefs)
    for subjectdef in root.xpath("./*[local-name()='subjectdef']"):
        parsed = parse_subject(subjectdef)
        if parsed:
            subjects.append(parsed)
            
    # Extract enumerations (Using .xpath() instead of .find() to support local-name())
    enumerations = []
    for enum_def in root.xpath(".//*[local-name()='enumerationdef']"):
        attr_elements = enum_def.xpath(".//*[local-name()='attributedef']")
        subj_elements = enum_def.xpath(".//*[local-name()='subjectdef']")
        
        if attr_elements and subj_elements:
            attr_name = attr_elements[0].get("name")
            keyref = subj_elements[0].get("keyref")
            if attr_name and keyref:
                enumerations.append({
                    "attribute": attr_name,
                    "subject_keyref": keyref
                })
                
    # Construct final target dictionary layout
    taxonomy_data = {
        "taxonomy": {
            "id": tax_id,
            "language": lang,
            "title": title_text,
            "subjects": subjects,
            "enumerations": enumerations
        }
    }
    
    # Ensure output directory exists
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(taxonomy_data, f, sort_keys=False, allow_unicode=True)
        
    print(f"Successfully converted {xml_path} to {yaml_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    xml_file = base_dir / "content" / "taxonomy" / "pharma-product-taxonomy.ditamap"
    yaml_file = base_dir / "schemas" / "pharma-product-taxonomy.yaml"
    
    convert_taxonomy(xml_file, yaml_file)