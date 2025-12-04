import re

input_path = "poetry_dataset_final_poems_formatted.txt"
output_path = "poetry_dataset_final_poems_finalstructured.txt"

def merge_split_caps(poem: str) -> str:
    """Merge wrongly split capital letters like:
       T\nI\nM\nE -> TIME
    """
    lines = [l.strip() for l in poem.splitlines() if l.strip()]
    merged_lines = []
    i = 0
    while i < len(lines):
        group = []
        while i < len(lines) and re.fullmatch(r"[A-Z][,;:.!?-]?", lines[i]):
            group.append(lines[i].rstrip(",;:.!?-"))
            i += 1
        if len(group) >= 3:  # likely a broken word
            merged_lines.append("".join(group))
        elif group:
            merged_lines.extend(group)
        if i < len(lines):
            merged_lines.append(lines[i])
            i += 1
    return "\n".join(merged_lines)

def clean_whitespace(poem: str) -> str:
    # Normalize spaces but keep line breaks
    poem = re.sub(r'[ \t]+', ' ', poem)
    poem = re.sub(r'\n{3,}', '\n\n', poem)  # limit blank lines
    return poem.strip()

with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

poems = text.split("<|endoftext|>")
fixed_poems = []

for raw_poem in poems:
    poem = raw_poem.strip()
    if not poem:
        continue
    poem = merge_split_caps(poem)
    poem = clean_whitespace(poem)
    fixed_poems.append(poem)

with open(output_path, "w", encoding="utf-8") as f:
    for p in fixed_poems:
        f.write(p.strip() + "\n\n<|endoftext|>\n\n")

print(f"✅ Saved structured poetry dataset to {output_path}")
print(f"Total poems retained: {len(fixed_poems)}")