# cleans up the messy label strings GroundingDINO returns.
# it concatenates caption-token subsets per detection, so a "can" detection
# might come back as "can bottle jar" or "basketon" or "##on package".
# we want one clean noun per detection.

import re
from typing import List, Optional, Set


# common little words GroundingDINO mixes in via its BERT tokenizer.
STOP_GLUE = {
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "with", "for",
    "is", "are", "this", "that", "these", "those", "it", "its",
}


def _split_stuck_tokens(token, vocab):
    # try to split things like "basketon" -> ["basket", "on"] by looking for a
    # prefix that's in our known vocab. recurse on the rest.
    if not token or token in vocab:
        return [token]
    for i in range(len(token) - 1, 1, -1):
        prefix = token[:i]
        suffix = token[i:]
        if prefix in vocab:
            return [prefix] + _split_stuck_tokens(suffix, vocab)
    return [token]


def clean_label(raw, prompt_vocab=None):
    # raw: a label string from GroundingDINO, e.g. "can bottle jar" or "basketon"
    # prompt_vocab: set of lowercase nouns from the prompt, used to un-stick
    #   tokens like "basketon". optional.
    # returns: cleaned label string, or "" if nothing usable.
    if not raw:
        return ""

    # 1. drop BERT subword markers like "##on".
    cleaned = re.sub(r'##\w*', '', raw).strip()

    # 2. lowercase + split on whitespace.
    tokens = cleaned.lower().split()

    # 3. un-stick tokens using vocab if we have it.
    if prompt_vocab:
        unstuck = []
        for t in tokens:
            unstuck.extend(_split_stuck_tokens(t, prompt_vocab))
        tokens = unstuck

    # 4. drop glue words and single chars.
    tokens = [t for t in tokens if t not in STOP_GLUE and len(t) > 1]

    # 5. dedupe but keep order.
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)

    return " ".join(out) if out else ""


def extract_prompt_vocab(prompt):
    # turn a GSAM2 prompt like "can. bottle. food package."
    # into a set of single-word lowercase tokens we recognize as content words.
    vocab = set()
    for phrase in prompt.split("."):
        for token in phrase.strip().lower().split():
            if len(token) > 1 and token not in STOP_GLUE:
                vocab.add(token)
    return vocab


def merge_detections_to_unique_regions(boxes, labels, confs, iou_threshold=0.5):
    # GroundingDINO often returns the same object multiple times with different
    # label subsets. cluster by spatial overlap (IoU) so each cluster is one
    # physical object; keep all labels but pick the highest-conf one as primary.
    # returns: [{"box": [x1,y1,x2,y2], "labels": [...], "confs": [...]}, ...]
    # sorted by best confidence descending.

    def iou(a, b):
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    # walk detections highest-conf first so the first one in each cluster is the best.
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


def best_label_for_cluster(cluster):
    # we sorted clusters by confidence, so the first label is the best one.
    # fall back to "object" if there's nothing usable.
    if not cluster["labels"]:
        return "object"
    best = cluster["labels"][0]
    return best if best else "object"


# quick demo of what cleaning looks like on a real GroundingDINO output.
if __name__ == "__main__":
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
