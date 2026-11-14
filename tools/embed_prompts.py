"""Builds assets/person_prompts.json: the text side of the zero-shot person classifier.

HalalifyVid decides who a detected person is by comparing a SigLIP 2 image
embedding of the person with text embeddings of two groups of descriptions.
The text embeddings never change, so they are computed once here and shipped
with the program; at run time only the image encoder is needed.

Run this again after editing the prompt lists:

    tools/.venv/Scripts/python tools/embed_prompts.py <dir with text_model_fp16.onnx and tokenizer.json>

The two files come from
https://huggingface.co/onnx-community/siglip2-base-patch16-256-ONNX
(onnx/text_model_fp16.onnx and tokenizer.json). Needs: onnxruntime, tokenizers, numpy.
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL = "siglip2-base-patch16-256"
TEMPLATE = "this is a photo of {}."
MAX_TOKENS = 64  # SigLIP 2 is trained with text padded to exactly 64 tokens
PAD_ID, EOS_ID = 0, 1

# The two lists mirror each other line by line, so that neither group is
# favoured by anything except the person's sex: every setting, age, garment
# and viewpoint mentioned for one is mentioned for the other.
GROUPS = {
    "female": [
        "a woman",
        "a girl",
        "a lady",
        "a young woman",
        "an old woman",
        "a female person",
        "a woman's face",
        "a woman seen from behind",
        "a woman with long hair",
        "a woman wearing a hijab",
        "a woman wearing a niqab",
        "a woman in an abaya",
        "a woman in a dress",
        "a businesswoman",
        "a female athlete",
        "an actress",
    ],
    "male": [
        "a man",
        "a boy",
        "a gentleman",
        "a young man",
        "an old man",
        "a male person",
        "a man's face",
        "a man seen from behind",
        "a man with short hair",
        "a man with a beard",
        "a man wearing a turban",
        "a man in a thobe",
        "a man in a suit",
        "a businessman",
        "a male athlete",
        "an actor",
    ],
}


def tokenize(tokenizer: Tokenizer, texts: list[str]) -> np.ndarray:
    ids = np.full((len(texts), MAX_TOKENS), PAD_ID, dtype=np.int64)
    for row, text in enumerate(texts):
        tokens = tokenizer.encode(text.lower(), add_special_tokens=False).ids[: MAX_TOKENS - 1]
        tokens.append(EOS_ID)
        ids[row, : len(tokens)] = tokens
    return ids


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main() -> None:
    source = Path(sys.argv[1])
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    session = ort.InferenceSession(str(source / "text_model_fp16.onnx"), providers=["CPUExecutionProvider"])
    output = {"model": MODEL, "template": TEMPLATE, "groups": {}}
    centres = {}
    for name, prompts in GROUPS.items():
        ids = tokenize(tokenizer, [TEMPLATE.format(p) for p in prompts])
        pooled = session.run(["pooler_output"], {"input_ids": ids})[0].astype(np.float64)
        centre = unit(unit(pooled).mean(axis=0))
        centres[name] = centre
        output["groups"][name] = {"prompts": prompts, "embedding": [round(float(x), 6) for x in centre]}
    output["dim"] = int(len(centres["female"]))

    destination = Path(__file__).resolve().parent.parent / "assets" / "person_prompts.json"
    destination.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {destination} (dim {output['dim']})")
    print(f"cosine(female, male) = {float(centres['female'] @ centres['male']):.4f}")


if __name__ == "__main__":
    main()
