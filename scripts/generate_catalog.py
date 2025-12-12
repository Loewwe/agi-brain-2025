
import re
import json
from pathlib import Path

def generate_catalog():
    md_path = Path("/Users/ongarkurmangaliyev/.gemini/antigravity/brain/be558f72-2221-4d6c-b45f-2a70a31ca988/HYPOTHESIS_CATALOG_126.md")
    output_path = Path("catalog/all_hypotheses.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(md_path, "r") as f:
        content = f.read()
        
    # Regex to find table rows: | ID | Name | ... | Feasible |
    # Example: | H001 | Asian Morning Pump Fade | Fade sharp moves... | ✅ True |
    # Example: | H031 | Liquidity Vacuum Gap Fill | ❌ False (Need Orderbook) |
    
    # We need to capture ID, Name, and Feasible status
    # The format varies slightly between sections (some have Logic column, some don't)
    
    hypotheses = []
    
    lines = content.split('\n')
    current_group = "Unknown"
    
    for line in lines:
        line = line.strip()
        
        # Detect Group
        if line.startswith("## "):
            # Example: ## 1. Basic / Session / Time (H001-H030)
            current_group = line.replace("## ", "").strip()
            
        if not line.startswith("| H"):
            continue
            
        parts = [p.strip() for p in line.split('|') if p.strip()]
        
        if len(parts) < 3:
            continue
            
        hyp_id = parts[0]
        name = parts[1]
        
        # Logic to extract feasible status
        # It's usually the last column
        feasible_col = parts[-1]
        feasible = "True" in feasible_col or "✅" in feasible_col
        
        # Extract logic if present (for H001-H030 it's column 2)
        logic = ""
        if len(parts) == 4 and "Logic" in lines[lines.index(line)-2]: # Heuristic check
             logic = parts[2]
        
        hypotheses.append({
            "id": hyp_id,
            "name": name,
            "group": current_group,
            "feasible": feasible,
            "description": logic
        })
        
    print(f"Found {len(hypotheses)} hypotheses.")
    
    with open(output_path, "w") as f:
        json.dump(hypotheses, f, indent=2)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_catalog()
