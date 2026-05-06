"""
Generic label cleaner for GroundingDINO output.

GroundingDINO returns labels as concatenated caption-token subsets per detection,
e.g. "can bottle jar", "basketon", "##on package". This module cleans them into
a single normalized noun per detection.

The cleaner is scene-agnostic — it doesn't know about specific objects.
It works by:
  1. Tokenizing on whitespace
  2. Dropping BERT subword junk (##xxx) and known glue words (on/the/a/an/in/of/etc)
  3. Joining stuck tokens like "basketon" back into "basket"
  4. Returning the first surviving content token, OR a multi-token phrase if length > 1

Caller then passes the original PROMPT to know what content tokens are valid.
"""
import re
from typing import List, Optional, Set


# minimal set of stopwords that GroundingDINO mixes into its detections
# These are non-content words that creep in via the BERT tokenizer
STOP_GLUE = {
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "with", "for",
    "is", "are", "this", "that", "these", "those", "it", "its",
}


def _split_stuck_tokens(token: str, vocab: Set[str]) -> List[str]:
    """
    'basketon' might be 'basket' + 'on'.
    Try to split a token by checking if any prefix is in our known vocab.
    Returns the list of tokens after splitting (single-element if no split found).
    """
    if not token or token in vocab:
        return [token]
    # try every prefix length from longest to shortest
    for i in range(len(token) - 1, 1, -1):
        prefix = token[:i]
        suffix = token[i:]
        if prefix in vocab:
            return [prefix] + _split_stuck_tokens(suffix, vocab)
    return [token]


def clean_label(raw: str, prompt_vocab: Optional[Set[str]] = None) -> str:
    """
    Clean a GroundingDINO label string.
    
    Args:
      raw: raw label string from X-Labels header, e.g. "can bottle jar" or "basketon"
      prompt_vocab: set of nouns from the prompt (lowercased), used to split
                    stuck tokens like "basketon" → ["basket", "on"]. If not provided,
                    tries simple cleaning only.
    
    Returns:
      cleaned label string. Empty string if nothing usable found.
    """
    if not raw:
        return ""
    
    # 1. drop BERT subword markers
    cleaned = re.sub(r'##\w*', '', raw).strip()
    
    # 2. tokenize
    tokens = cleaned.lower().split()
    
    # 3. try to split stuck tokens like "basketon" using vocab if provided
    if prompt_vocab:
        unstuck = []
        for t in tokens:
            unstuck.extend(_split_stuck_tokens(t, prompt_vocab))
        tokens = unstuck
    
    # 4. drop glue words
    tokens = [t for t in tokens if t not in STOP_GLUE and len(t) > 1]
    
    # 5. dedupe but preserve order
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    
    return " ".join(out) if out else ""


def extract_prompt_vocab(prompt: str) -> Set[str]:
    """
    Build a vocabulary set from a GSAM2 prompt string.
    Prompt format: "thing1. thing2. thing with multiple words. ..."
    Returns lowercased single-word tokens that appeared in the prompt.
    """
    vocab = set()
    for phrase in prompt.split("."):
        for token in phrase.strip().lower().split():
            if len(token) > 1 and token not in STOP_GLUE:
                vocab.add(token)
    return vocab


def merge_detections_to_unique_regions(
    boxes: List[List[float]],
    labels: List[str],
    confs: List[float],
    iou_threshold: float = 0.5,
):
    """
    Merge overlapping GroundingDINO detections into unique object regions.
    
    GroundingDINO often returns the same object multiple times with different
    label subsets. This function clusters detections by spatial overlap (IoU)
    and keeps the highest-confidence per cluster, while collecting all labels.
    
    Args:
      boxes: list of [x1, y1, x2, y2] for each detection
      labels: list of cleaned label strings (same length as boxes)
      confs: list of confidence floats (same length as boxes)
      iou_threshold: detections with IoU above this are merged
    
    Returns:
      list of dicts: [{"box": [...], "labels": [str, ...], "confs": [...]}, ...]
      sorted by best-confidence descending.
    """
    def iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        return inter / (area_a + area_b - inter)
    
    # sort by confidence desc
    order = sorted(range(len(boxes)), key=lambda i: -confs[i])
    
    clusters = []
    for i in order:
        merged = False
        for c in clusters:
            if iou(boxes[i], c["box"]) > iou_threshold:
                c["labels"].append(labels[i])
                c["confs"].append(confs[i])
                merged = True
                break
        if not merged:
            clusters.append({
                "box": boxes[i],
                "labels": [labels[i]],
                "confs": [confs[i]],
            })
    
    return clusters


def best_label_for_cluster(cluster: dict) -> str:
    """
    Given a cluster of (box, [labels], [confs]), pick the most informative label.
    Strategy: take the label from the highest-conf detection.
    """
    if not cluster["labels"]:
        return "object"
    # highest-conf is at index 0 by construction in merge_detections
    best = cluster["labels"][0]
    return best if best else "object"


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    # simulated GroundingDINO output from libero_object scene
    prompt = "can. box. bottle. basket. carton. jar. food package. small object."
    vocab = extract_prompt_vocab(prompt)
    print(f"Prompt vocabulary: {sorted(vocab)}\n")
    
    raw_labels = [
        "basketon",
        "bottle jar",
        "small object",
        "can bottle jar",
        "can jar",
        "carton food package",
        "boxon food package small object",
        "##on package",
        "jar",
    ]
    print(f"{'raw':>40}  {'cleaned':>30}")
    print("-" * 75)
    for r in raw_labels:
        c = clean_label(r, vocab)
        print(f"{r:>40}  {c:>30}")
