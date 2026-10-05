import json
import lxml.etree as etree  # type: ignore
from pathlib import Path

# Anchors all paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CONTENT_DIR = PROJECT_ROOT / "content" / "topics"
DITAVAL_PATH = PROJECT_ROOT / "config" / "ditaval-filters.ditaval"
TAXONOMY_PATH = PROJECT_ROOT / "content" / "taxonomy" / "pharma-product-taxonomy.ditamap"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dita-processed"


def load_ditaval_rules(ditaval_file: Path) -> list[tuple[str, str]]:
    """Parses config/ditaval-filters.ditaval for action='exclude' rules."""
    exclude_rules = []
    if not ditaval_file.exists():
        print(f"Warning: {ditaval_file} not found. Proceeding without filtering.")
        return exclude_rules

    tree = etree.parse(str(ditaval_file))
    root = tree.getroot()

    for prop in root.findall(".//prop[@action='exclude']"):
        att = prop.get("att")
        val = prop.get("val")
        if att and val:
            exclude_rules.append((att, val))

    return exclude_rules


def parse_taxonomy_hierarchy(taxonomy_file: Path) -> dict[str, list[str]]:
    """
    Dynamically parses pharma-product-taxonomy.ditamap to build parent-child
    lineage mappings directly from the subjectScheme XML structure.
    """
    hierarchy_map = {}
    if not taxonomy_file.exists():
        print(f"Warning: Taxonomy map {taxonomy_file} not found.")
        return hierarchy_map

    tree = etree.parse(str(taxonomy_file))
    root = tree.getroot()

    # Traverses subjectdef nodes recursively to build parent chains
    def traverse_subject(elem, parent_chain):
        key = elem.get("keys")
        if key:
            if parent_chain:
                hierarchy_map[key] = list(parent_chain)
            current_chain = [key] + parent_chain
        else:
            current_chain = parent_chain

        for child in elem.findall("subjectdef"):
            traverse_subject(child, current_chain)

    for top_subject in root.findall("subjectdef"):
        traverse_subject(top_subject, [])

    return hierarchy_map


def expand_taxonomy_lineage(raw_attr_value: str, taxonomy_map: dict[str, list[str]]) -> list[str]:
    """Splits space-separated attribute tokens and adds parent terms from the taxonomy."""
    tokens = raw_attr_value.split()
    expanded = set(tokens)

    for token in tokens:
        parents = taxonomy_map.get(token, [])
        expanded.update(parents)

    return sorted(list(expanded))


def should_exclude(element, exclude_rules: list[tuple[str, str]]) -> bool:
    """Checks space-separated attributes against exclude rules."""
    for att, val in exclude_rules:
        raw_attr_val = element.get(att, "")
        tokens = raw_attr_val.split()
        if val in tokens:
            return True
    return False


def apply_ditaval_filtering(root, exclude_rules: list[tuple[str, str]]):
    """Prunes matching subtrees directly from the DOM before text extraction."""
    for elem in list(root.iter()):
        if should_exclude(elem, exclude_rules):
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)


def extract_table_text(table_elem) -> str:
    """Flattens DITA <table> elements into plain text tables."""
    rows_text = []
    headers = [th.text.strip() for th in table_elem.findall(".//thead//entry") if th.text]
    if headers:
        rows_text.append(" | ".join(headers))

    for row in table_elem.findall(".//tbody/row"):
        entries = ["".join(entry.itertext()).strip() for entry in row.findall("entry")]
        if entries:
            rows_text.append(" | ".join(entries))

    return "\n".join(rows_text)


def extract_node_text(node) -> str:
    """Extracts text from paragraphs, lists, tables, and task steps."""
    text_chunks = []

    for elem in node:
        if elem.tag == "title":
            continue

        if elem.tag == "table":
            tbl_txt = extract_table_text(elem)
            if tbl_txt:
                text_chunks.append(tbl_txt)

        elif elem.tag in ("ul", "ol"):
            items = ["- " + "".join(li.itertext()).strip() for li in elem.findall("li")]
            if items:
                text_chunks.append("\n".join(items))

        elif elem.tag == "steps":
            steps_text = []
            for idx, step in enumerate(elem.findall("step"), 1):
                cmd = step.findtext("cmd") or ""
                xmp = step.findtext("stepxmp") or ""
                info = step.findtext("info") or ""
                result = step.findtext("stepresult") or ""

                step_str = f"Step {idx}: {cmd}".strip()
                if xmp:
                    step_str += f" (Example: {xmp})"
                if info:
                    step_str += f" Note: {info}"
                if result:
                    step_str += f" Result: {result}"
                steps_text.append(step_str)

            if steps_text:
                text_chunks.append("\n".join(steps_text))

        else:
            txt = "".join(elem.itertext()).strip()
            if txt:
                text_chunks.append(txt)

    return "\n".join(text_chunks)


def parse_dita_topic(file_path: Path, exclude_rules: list[tuple[str, str]], taxonomy_map: dict[str, list[str]]) -> dict:
    """Parses DITA topics into structured JSON with taxonomy lineage expansion."""
    tree = etree.parse(str(file_path))
    root = tree.getroot()

    # Prune DITAVAL exclusions
    apply_ditaval_filtering(root, exclude_rules)

    title = root.findtext("title") or ""
    shortdesc = root.findtext("shortdesc") or ""

    raw_audience = root.get("audience", "")
    raw_product = root.get("product", "")
    raw_otherprops = root.get("otherprops", "")

    # Dynamically expands hierarchy using parsed taxonomy
    audience_lineage = expand_taxonomy_lineage(raw_audience, taxonomy_map)
    intent_lineage = expand_taxonomy_lineage(raw_otherprops, taxonomy_map)

    sections = []
    body = root.find("refbody") or root.find("conbody") or root.find("taskbody")

    if body is not None:
        direct_text = extract_node_text(body)
        if direct_text:
            sections.append({"section_title": "Overview & Context", "content": direct_text})

        for sec in body.findall("section"):
            sec_title = sec.findtext("title") or "Section"
            sec_text = extract_node_text(sec)
            if sec_text:
                sections.append({"section_title": sec_title, "content": sec_text})

    return {
        "doc_id": file_path.stem,
        "title": title,
        "shortdesc": shortdesc,
        "sections": sections,
        "metadata": {
            "source_file": file_path.name,
            "category": file_path.parent.name,
            "product": raw_product,
            "audience": ", ".join(audience_lineage),
            "otherprops": ", ".join(intent_lineage),
            "audience_list": audience_lineage,
            "otherprops_list": intent_lineage,
        },
    }


def process_all_topics(dry_run: bool = False):
    exclude_rules = load_ditaval_rules(DITAVAL_PATH)
    taxonomy_map = parse_taxonomy_hierarchy(TAXONOMY_PATH)

    print(f"Loaded {len(exclude_rules)} exclusion rules from DITAVAL.")
    print(f"Parsed {len(taxonomy_map)} dynamic taxonomy lineages from subjectScheme.\n")

    for dita_file in CONTENT_DIR.rglob("*.dita"):
        data = parse_dita_topic(dita_file, exclude_rules, taxonomy_map)

        if dry_run:
            print("=" * 80)
            print(f"PREVIEW: {dita_file.name}")
            print("=" * 80)
            print(json.dumps(data, indent=2))
            print("\n")
        else:
            rel_path = dita_file.relative_to(CONTENT_DIR)
            out_file = OUTPUT_DIR / rel_path.parent / f"{dita_file.stem}.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            print(f"Processed: {dita_file.name} -> {out_file}")


if __name__ == "__main__":
    # Set False to generate output files in data/dita-processed/
    DRY_RUN_MODE = True
    process_all_topics(dry_run=DRY_RUN_MODE)