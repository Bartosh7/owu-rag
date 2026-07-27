import os
import re
from collections import Counter

# Wyrażenia z `.*` (łapią wszystko), ale bezpieczne dzięki \b (całe słowa) i limitowi długości linii
VISUAL_NOISE_PATTERNS = [
    ("logo", re.compile(r"^\s*.*\blogo\b.*\s*$", re.IGNORECASE)),
    ("icon", re.compile(r"^\s*.*\bicon\b.*\s*$", re.IGNORECASE)),
    ("ikona", re.compile(r"^\s*.*\bikona\b.*\s*$", re.IGNORECASE)),
    ("qr code", re.compile(r"^\s*.*\bqr\s+code\b.*\s*$", re.IGNORECASE)),
    ("kod qr", re.compile(r"^\s*.*\bkod\s+qr\b.*\s*$", re.IGNORECASE)),
    ("separator", re.compile(r"^\s*---\s*$")),
    ("strzalka", re.compile(r"^\s*.*\bstrza[lł]ka\b.*\s*$", re.IGNORECASE)),
]


def remove_specific_pages(text: str, pages_to_remove: list) -> str:
    """Usuwa zawartość podanych stron na podstawie znacznika <!-- PAGE: X -->"""
    # Dodałem \s* na wypadek jakichkolwiek nadmiarowych spacji wokół numeru
    pages_raw = re.split(r'(<!--\s*PAGE:\s*\d+\s*-->)', text)
    filtered_parts = []
    skip_next = False
    
    for part in pages_raw:
        match = re.match(r'<!--\s*PAGE:\s*(\d+)\s*-->', part)
        if match:
            page_num = int(match.group(1))
            if page_num in pages_to_remove:
                skip_next = True
            else:
                skip_next = False
                filtered_parts.append(part)
        else:
            if not skip_next:
                filtered_parts.append(part)
                
    return "".join(filtered_parts)


def remove_visual_noise(text: str):
    lines = text.splitlines()

    cleaned = []
    removed = []
    stats = Counter()

    for line in lines:
        stripped = line.strip()
        should_remove = False

        # TARCZA OCHRONNA: Sprawdzamy tylko linie krótsze niż 40 znaków.
        if len(stripped) < 40:
            for category, pattern in VISUAL_NOISE_PATTERNS:
                if pattern.match(stripped):
                    should_remove = True
                    stats[category] += 1
                    removed.append(stripped)
                    break

        if not should_remove:
            cleaned.append(line)

    return "\n".join(cleaned), removed, stats


def normalize_markdown(text):
    print("[INFO] Running Markdown normalization...")

    # --------------------------------------------------------
    # RULE 0 - Remove title page and table of contents
    # --------------------------------------------------------
    pages_to_skip = [1, 6, 7]
    print(f"[INFO] Removing pages: {pages_to_skip} (Title page, Table of contents)...")
    
    text = remove_specific_pages(text, pages_to_skip)

    # --------------------------------------------------------
    # RULE 1 - Normalize paragraph headers
    # --------------------------------------------------------
    text = re.sub(
        r"^#+\s*§",
        "## §",
        text,
        flags=re.MULTILINE,
    )

    # --------------------------------------------------------
    # RULE 2 - Remove orphan page numbers
    # --------------------------------------------------------
    text = re.sub(
        r"^\d+\s*\n+---\n+",
        "---\n",
        text,
        flags=re.MULTILINE,
    )

    # --------------------------------------------------------
    # RULE 3 - Remove obvious visual noise
    # --------------------------------------------------------
    text, removed, stats = remove_visual_noise(text)

    print(f"\n[INFO] Removed visual-noise lines: {len(removed)}")

    if removed:
        print("\nRemoved lines:")
        for line in removed:
            print(f"  • {line}")

    if stats:
        print("\nSummary:")
        for category, count in stats.items():
            print(f"  {category:<10}: {count}")

    return text


def main():
    input_path = "data/processed/parsed_owu.md"
    output_path = "data/processed/cleaned_owu.md"

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"[ERROR] Input file not found: {input_path}"
        )

    print("[INFO] Loading parsed Markdown...")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_markdown = f.read()

    cleaned_markdown = normalize_markdown(raw_markdown)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_markdown)

    print(f"\n[SUCCESS] Normalization complete. Cleaned file saved to:\n{output_path}")


if __name__ == "__main__":
    main()