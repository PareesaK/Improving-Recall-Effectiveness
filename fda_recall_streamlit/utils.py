import re

def clean_text(text):
    """Clean and normalize text data."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase 
    text = str(text).lower()
    
    # Normalize pathogen names
    text = re.sub(r'listeria\s+monocytogenes', 'listeria_monocytogenes', text)
    text = re.sub(r'escherichia\s+coli', 'ecoli', text)
    text = re.sub(r'e\.?\s*coli', 'ecoli', text)
    text = re.sub(r'salmonella\s+\w+', 'salmonella', text)
    
    # Normalize allergen references
    text = re.sub(r'undeclared\s+(milk|soy|wheat|egg|peanut|tree\s+nut|fish|shellfish)', 
                 r'undeclared_allergen_\1', text)
    
    # Normalize manufacturing issues
    text = re.sub(r'(manufacturing|production)\s+(defect|error|issue|problem)', 
                 'manufacturing_defect', text)
    
    # Remove special characters but preserve normalized terms
    text = re.sub(r'[^\w\s_]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_recall_flags(reason_text):
    """Extract specific recall indicators from text."""
    flags = {}
    reason = str(reason_text).lower() if isinstance(reason_text, str) else ""
    
    # Pathogens
    flags['has_listeria'] = 1 if re.search(r'listeria|monocytogenes', reason) else 0
    flags['has_salmonella'] = 1 if 'salmonella' in reason else 0
    flags['has_ecoli'] = 1 if re.search(r'e\.?\s*coli|escherichia', reason) else 0
    
    # Allergens
    flags['has_undeclared'] = 1 if 'undeclared' in reason else 0
    flags['has_allergen'] = 1 if re.search(r'allergen|allergic|allergy', reason) else 0
    for allergen in ['milk', 'soy', 'wheat', 'egg', 'peanut', 'nut', 'fish', 'shellfish']:
        flags[f'allergen_{allergen}'] = 1 if allergen in reason else 0
    
    # Manufacturing issues
    flags['has_manufacturing_issue'] = 1 if re.search(r'manufactur|production|process', reason) else 0
    flags['has_quality_issue'] = 1 if re.search(r'quality|control|inspection', reason) else 0
    flags['has_mislabeling'] = 1 if re.search(r'mislabel|incorrect label|wrong label', reason) else 0
    
    # Foreign material
    flags['has_foreign_material'] = 1 if re.search(r'foreign|material|particulate|metal|glass|plastic', reason) else 0
    
    # Risk indicators
    flags['possible_illness'] = 1 if re.search(r'illness|disease|sick|adverse', reason) else 0
    flags['possible_injury'] = 1 if re.search(r'injur|harm|wound', reason) else 0
    
    return flags

def categorize_distribution(pattern):
    """Simplify Distribution Pattern into categories."""
    if not isinstance(pattern, str):
        return "Unknown"
    
    pattern = pattern.lower()
    
    if any(term in pattern for term in ['nationwide', 'national', 'across us', 'throughout us']):
        return 'Nationwide'
    elif any(term in pattern for term in ['worldwide', 'international', 'global', 'exported']):
        return 'International'
    elif any(term in pattern for term in ['regional', 'multi-state', 'states', 'multiple states']):
        return 'Regional'
    elif any(term in pattern for term in ['limited', 'local', 'single']):
        return 'Limited'
    else:
        return 'Other'