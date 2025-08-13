import pathlib, chardet, os
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
ROOT = pathlib.Path('.knowledge')
def ensure_dirs():
    (ROOT/"indexes"/"chunks").mkdir(parents=True, exist_ok=True)
    (ROOT/"indexes"/"embeddings").mkdir(parents=True, exist_ok=True)
    (ROOT/"indexes"/"graph").mkdir(parents=True, exist_ok=True)
    (ROOT/"indexes"/"summaries").mkdir(parents=True, exist_ok=True)
    (ROOT/"indexes"/"attributes").mkdir(parents=True, exist_ok=True)
    (ROOT/"indexes"/"manifests").mkdir(parents=True, exist_ok=True)
    (ROOT/"queues").mkdir(parents=True, exist_ok=True)
    (ROOT/"exports"/"dumps").mkdir(parents=True, exist_ok=True)
def read_text_safely(path: pathlib.Path):
    try:
        suffix = path.suffix.lower()
        if suffix == '.pdf' and fitz is not None:
            text_parts = []
            with fitz.open(str(path)) as doc:
                for page in doc:
                    text_parts.append(page.get_text("text"))
            txt = "\n".join(text_parts)
            if txt.strip():
                return txt
            # fall through to OCR if enabled
        # OCR for images (or empty PDFs) when enabled
        if (suffix in {'.png','.jpg','.jpeg','.tif','.tiff'} or (suffix=='.pdf' and fitz is not None)) and (os.getenv('OCR_ENABLED','false').lower()=='true') and pytesseract and Image:
            tcmd = os.getenv('TESSERACT_CMD') or ""
            if tcmd:
                try: pytesseract.pytesseract.tesseract_cmd = tcmd
                except Exception: pass
            try:
                if suffix=='.pdf' and fitz is not None:
                    # render pages to images and OCR
                    out = []
                    with fitz.open(str(path)) as doc:
                        for page in doc:
                            pix = page.get_pixmap(dpi=200)
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            out.append(pytesseract.image_to_string(img))
                    ocr_txt = "\n".join(out)
                else:
                    ocr_txt = pytesseract.image_to_string(Image.open(str(path)))
                if ocr_txt.strip():
                    return ocr_txt
            except Exception:
                pass
        # default: try bytes decode with chardet
        data = path.read_bytes()
        enc = chardet.detect(data).get('encoding') or 'utf-8'
        return data.decode(enc, errors='ignore')
    except Exception:
        return None