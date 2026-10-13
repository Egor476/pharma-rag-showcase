import json
from jsonschema import validate
from pathlib import Path

def validate_task_prompts():
    base_dir = Path(__file__).resolve().parent.parent
    schema_path = base_dir / "prompt-library" / "schemas" / "prompt-schema.json"
    prompts_dir = base_dir / "prompt-library" / "task-prompts"
    
    if not schema_path.exists():
        print(f"Error: Schema not found at {schema_path}")
        return False
        
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
        
    success = True
    print("Starting prompt schema validation...")
    
    for json_file in prompts_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                instance = json.load(f)
            validate(instance=instance, schema=schema)
            print(f"  [PASS] {json_file.name}")
        except Exception as e:
            print(f"  [FAIL] {json_file.name} failed validation: {e}")
            success = False
            
    return success

if __name__ == "__main__":
    import sys
    if not validate_task_prompts():
        sys.exit(1)
    else:
        print("All task prompts successfully validated!")
        sys.exit(0)