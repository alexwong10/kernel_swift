"""Extract the ten authoritative reference programs from 赛道说明.md.

The competition document contains executable Python directly after each TaskXX
heading. Keeping extraction mechanical avoids hand-copy drift when the document
is updated.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "赛道说明.md"
OUT_DIR = ROOT / "reference"

NAMES = {
    1: "grouped_topk",
    2: "fused_moe",
    3: "flex_attention",
    4: "splade_sparse_pooler",
    5: "music_flamingo_rotary_embedding",
    6: "mm_encoder_attention",
    7: "mhc_post",
    8: "hc_split_sinkhorn",
    9: "centre_random_augmentation",
    10: "head_compute_mix_bwd",
}


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    pattern = re.compile(r"^Task(\d{2})：[^\n]*\n", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if [int(m.group(1)) for m in matches] != list(range(1, 11)):
        raise RuntimeError("expected Task01..Task10 exactly once")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, match in enumerate(matches):
        task_no = int(match.group(1))
        start = match.end()
        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            marker = text.find("\n4. 作品提交方式", start)
            if marker < 0:
                raise RuntimeError("could not find end of Task10")
            end = marker
        code = text[start:end].strip() + "\n"
        path = OUT_DIR / f"{task_no:02d}_{NAMES[task_no]}.py"
        path.write_text(code, encoding="utf-8", newline="\n")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
