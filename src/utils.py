#Helper functions
# ----------------------------------------------------------------------------
def normalize_col(c: str) -> str:
    """Trim spaces, remove BOM/nbsp, collapse whitespace."""
    c = (c or "")
    c = c.replace("\ufeff", "").replace("\xa0", " ")
    c = re.sub(r"\s+", " ", c).strip()
    return c

def find_label_col(cols, name_target="label"):
    """Find Label column even if spacing/case differs."""
    for c in cols:
        if normalize_col(c).lower() == name_target.lower():
            return c
    for c in cols:
        if name_target.lower() in normalize_col(c).lower():
            return c
    return None

def is_dos_ddos(label: str) -> bool:
    """Check if label is DoS/DDoS related."""
    if pd.isna(label):
        return False
    label_lower = str(label).lower()
    dos_keywords = ['dos', 'ddos', 'hulk', 'goldeneye', 'slowloris', 'slowhttptest']
    return any(keyword in label_lower for keyword in dos_keywords)

print(" Helper functions defined")
