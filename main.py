import argparse, os, sys, io, json
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def extract_text_with_fallback(pdf_path, ocr_lang="deu+eng", ocr_threshold_chars=40):
    """
    1) Versuche nativen Text mit pypdf.
    2) Falls Seite (nahezu) leer -> OCR (Tesseract) über gerenderte Seite.
    """
    reader = PdfReader(pdf_path)
    texts = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        raw = raw.strip()
        if len(raw) >= ocr_threshold_chars:
            texts.append(raw)
            continue
        # Fallback OCR für diese Seite
        # Rendern mit 300 DPI; benötigt poppler-utils (pdftoppm)
        images = convert_from_path(pdf_path, dpi=300, first_page=i+1, last_page=i+1)
        if images:
            txt = pytesseract.image_to_string(images[0], lang=ocr_lang) or ""
            texts.append(txt.strip())
        else:
            texts.append(raw)
    return "\n\n".join(texts).strip()

def load_model(model_id="Qwen/Qwen2.5-7B-Instruct", device="cuda"):
    """
    Lädt ein kompaktes Instruct-Modell.
    - Passt auf gängige 16–24 GB GPUs (FP16).
    - Für knappere GPUs: 4-bit (bitsandbytes) ließe sich ergänzen.
    """
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return tok, model

def chat(tokenizer, model, system_prompt, user_prompt, max_new_tokens=800):
    tpl = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n" \
          f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n" \
          f"<|im_start|>assistant\n"
    inputs = tokenizer(tpl, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            streamer=streamer
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Pfad zur PDF-Datei (in Lyceum meist /lyceum/storage/...)")
    ap.add_argument("--prompt", required=True, help="Anweisung/Frage für die Analyse")
    ap.add_argument("--ocr-lang", default="deu+eng", help="Tesseract Sprachpakete (installiert: deu, eng)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HF Modell-ID")
    ap.add_argument("--out", default="/lyceum/storage/result.txt", help="Pfad für Ergebnisdatei")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        print(f"PDF nicht gefunden: {args.pdf}", file=sys.stderr)
        sys.exit(2)

    print(">> PDF analysieren …", file=sys.stderr)
    text = extract_text_with_fallback(args.pdf, ocr_lang=args.ocr_lang)

    if not text.strip():
        print("Leere PDF/Eingabe – Abbruch.", file=sys.stderr)
        sys.exit(3)

    system = (
        "Du bist ein gewissenhafter Assistent. "
        "Du erhältst den Textinhalt einer PDF (ggf. via OCR extrahiert). "
        "Analysiere präzise, zitiere relevante Passagen (mit Seitenhinweis, wenn möglich), "
        "und weise auf Unsicherheiten durch OCR hin."
    )
    user = f"PROMPT:\n{args.prompt}\n\n---\nPDF-TEXT:\n{text[:120000]}"

    print("\n>> LLM laden …", file=sys.stderr)
    tok, mdl = load_model(args.model)

    print("\n>> Antworte …\n", file=sys.stderr)
    # Streamt zur Konsole (Lyceum zeigt stdout live)
    chat(tok, mdl, system, user, max_new_tokens=1200)

    # Zusätzlich in Datei schreiben
    # (Lyceum: alles unter /lyceum/storage/ kannst du nach dem Job herunterladen)
    with open(args.out, "w", encoding="utf-8") as f:
        # Für die Datei generieren wir die Antwort noch einmal ohne Stream
        inputs = tok(
            f"<|im_start|>system\n{system}\n<|im_end|>\n"
            f"<|im_start|>user\n{user}\n<|im_end|>\n"
            f"<|im_start|>assistant\n", return_tensors="pt"
        ).to(mdl.device)
        out_ids = mdl.generate(
            **inputs, max_new_tokens=1200, do_sample=True, temperature=0.2, top_p=0.9
        )
        text_out = tok.decode(out_ids[0], skip_special_tokens=True)
        # Nur den Assistant-Teil grob extrahieren
        ans = text_out.split("<|im_start|>assistant")[-1].strip()
        f.write(ans)

    print(f"\n\n[GESPEICHERT] {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
