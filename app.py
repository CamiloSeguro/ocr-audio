import os
import io
import glob
import time
import base64
from datetime import datetime


import cv2
import numpy as np
import streamlit as st
import pytesseract
from PIL import Image
from gtts import gTTS
from googletrans import Translator

# ────────────────────────────── Config & Theme ────────────────────────────── #
st.set_page_config(
page_title="OCR + Traducción + TTS",
page_icon="🧠",
layout="wide",
initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
--bg: #f9f9fb;
--panel: #ffffff;
--text: #222222;
--muted: #555555;
--brand: #7c5cff;
}
/* Fondo y tipografía */
section[data-testid="stSidebar"] {background: var(--panel);}
.block-container {padding-top: 1.2rem; background: var(--bg);}
html, body, [class^="css"] { color: var(--text) !important; }
/* Tarjetas */
.card { background: var(--panel); border: 1px solid rgba(0,0,0,0.08); padding: 18px; border-radius: 16px; }
/* Botones */
.stButton > button { background: var(--brand) !important; color: white !important; border-radius: 12px; border: none; }
/* Inputs */
.stTextArea textarea, .stTextInput input { background: #f1f1f5; color: #111; border-radius: 12px; }
/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: .5rem; }
.stTabs [data-baseweb="tab"] { background: var(--panel); border-radius: 12px; color: var(--text); }
/* Alerts (fix recorte y bordes) */
div[data-testid="stAlert"] { border-radius: 12px !important; overflow: visible !important; }
div[data-testid="stAlert"] > div { border-radius: 12px !important; }
div[data-testid="stAlert"] p { margin: 0 !important; }
/* Badges */
.badge { display:inline-flex; align-items:center; gap:.5rem; padding:.35rem .6rem; border:1px solid rgba(0,0,0,.08); border-radius:999px; font-size:.8rem; color:var(--muted); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ────────────────────────────── Utils ────────────────────────────── #
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
translator = Translator()


LANG_MAP = {
"Inglés": "en",
"Español": "es",
"Bengalí": "bn",
"Coreano": "ko",
"Mandarín": "zh-cn",
"Japonés": "ja",
}


TLD_MAP = {
"Default": "com",
"India": "co.in",
"United Kingdom": "co.uk",
"United States": "com",
"Canada": "ca",
"Australia": "com.au",
"Ireland": "ie",
"South Africa": "co.za",
}


def remove_old_files(days: int = 7):
now = time.time()
for f in glob.glob(os.path.join(TEMP_DIR, "*.mp3")):
try:
if os.stat(f).st_mtime < now - days * 86400:
os.remove(f)
except Exception:
pass


remove_old_files()


@st.cache_data(show_spinner=False)
def bytes_to_cv2_image(file_bytes: bytes) -> np.ndarray:
return cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)


@st.cache_data(show_spinner=False)
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


@st.cache_data(show_spinner=False)
def preprocess_image(img_bgr: np.ndarray, *, grayscale: bool, invert: bool, thresh: bool, blur_ksize: int) -> np.ndarray:
img = img_bgr.copy()
if grayscale:
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
if blur_ksize and blur_ksize % 2 == 1:
img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
if thresh:
if img.ndim == 3:
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
if invert:
img = cv2.bitwise_not(img)
# Asegurar salida RGB para mostrar y para Tesseract (que acepta BGR/GRAY igualmente)
if img.ndim == 2:
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
else:
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
return img_rgb


@st.cache_data(show_spinner=False)
def ocr_extract(img_rgb: np.ndarray, tess_lang: str = "eng"):
# Texto simple
text = pytesseract.image_to_string(img_rgb, lang=tess_lang)
# Métricas
data = pytesseract.image_to_data(img_rgb, lang=tess_lang, output_type=pytesseract.Output.DICT)
confs = [int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) >= 0]
mean_conf = float(np.mean(confs)) if confs else 0.0
return text, mean_conf


def text_to_speech(input_language: str, output_language: str, text: str, tld: str):
translation = translator.translate(text or "", src=input_language, dest=output_language)
trans_text = translation.text
safe_stub = (trans_text.strip() or "audio").replace("\n", " ")[:32]
filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_stub}.mp3"
out_path = os.path.join(TEMP_DIR, filename)
tts = gTTS(trans_text, lang=output_language, tld=tld, slow=False)
tts.save(out_path)
return out_path, trans_text

# ────────────────────────────── Sidebar ────────────────────────────── #
with st.sidebar:
    st.markdown("### 🛠️ Controles")

    st.session_state.ui_theme = st.radio(
        "Tema de la interfaz",
        ["Claro", "Oscuro"],
        index=0 if st.session_state.ui_theme == "Claro" else 1,
        horizontal=True,
    )
    inject_theme(st.session_state.ui_theme)

    source = st.radio("Fuente de imagen", ["Cámara", "Subir archivo"], horizontal=True)

    st.markdown("#### 🎛️ Preprocesamiento")
    c1, c2, c3 = st.columns(3)
    grayscale = c1.toggle("Grises", value=True)
    invert = c2.toggle("Invertir", value=False)
    thresh = c3.toggle("Umbral", value=True)
    blur_ksize = st.slider("Desenfoque (Gauss)", min_value=0, max_value=15, value=1, step=2, help="Kernel impar (0 desactiva)")

    st.markdown("#### 🔤 Idioma OCR (Tesseract)")
    tess_choice = st.selectbox("Selecciona idioma para OCR", ["Español (spa)", "Inglés (eng)", "Inglés+Español (eng+spa)"])
    tess_lang = {"Español (spa)": "spa", "Inglés (eng)": "eng", "Inglés+Español (eng+spa)": "eng+spa"}[tess_choice]

    st.markdown("#### 🌐 Traducción & Voz")
    in_lang_label = st.selectbox("Lenguaje de entrada", list(LANG_MAP.keys()), index=1)
    out_lang_label = st.selectbox("Lenguaje de salida", list(LANG_MAP.keys()), index=0)
    tld_label = st.selectbox("Acento de inglés (TLD)", list(TLD_MAP.keys()), index=0)
    show_output_text = st.checkbox("Mostrar texto traducido", value=True)

# ────────────────────────────── Header ────────────────────────────── #
left, right = st.columns([1, 1])
with left:
    st.markdown("# 🧠 OCR · Traducción · TTS")
    st.markdown("<span class='badge'>Rápido</span> <span class='badge'>Accesible</span> <span class='badge'>Multi‑idioma</span>", unsafe_allow_html=True)
with right:
    st.info("Consejo: mejora la precisión con **Grises + Umbral** y buena iluminación.")

# ────────────────────────────── Tabs ────────────────────────────── #
tab_capture, tab_ocr, tab_tts, tab_about = st.tabs(["📷 Captura", "🔎 OCR", "🔁 Traducción & Audio", "ℹ️ Acerca de"])

# Session state
if "latest_image" not in st.session_state:
    st.session_state.latest_image = None
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "ocr_conf" not in st.session_state:
    st.session_state.ocr_conf = 0.0

# ────────────────────────────── Captura ────────────────────────────── #
with tab_capture:
    st.markdown("### Fuente")

    file_bytes = None
    if source == "Cámara":
        cam = st.camera_input("Toma una foto", label_visibility="visible")
        if cam is not None:
            file_bytes = cam.getvalue()
    else:
        up = st.file_uploader("Cargar imagen", type=["png", "jpg", "jpeg", "webp"])
        if up is not None:
            file_bytes = up.read()

    colA, colB = st.columns([1, 1])
    if file_bytes:
        with st.spinner("Procesando imagen…"):
            img_bgr = bytes_to_cv2_image(file_bytes)
            img_rgb = preprocess_image(img_bgr, grayscale=grayscale, invert=invert, thresh=thresh, blur_ksize=blur_ksize)
            st.session_state.latest_image = img_rgb
        with colA:
            st.markdown("**Vista previa**")
            st.image(img_rgb, channels="RGB", use_container_width=True)
        with colB:
            st.markdown("**Acciones**")
            if st.button("Ejecutar OCR", use_container_width=True):
                with st.spinner("Leyendo texto…"):
                    text, conf = ocr_extract(st.session_state.latest_image, tess_lang=tess_lang)
                    st.session_state.ocr_text = text
                    st.session_state.ocr_conf = conf
                    st.success(f"OCR listo · confianza media: {conf:.1f}%")
    else:
        st.warning("Sube una imagen o usa la cámara para continuar.")

# ────────────────────────────── OCR ────────────────────────────── #
with tab_ocr:
    st.markdown("### Resultado OCR")
    if st.session_state.latest_image is not None and not st.session_state.ocr_text:
        if st.button("Ejecutar OCR ahora", type="primary"):
            with st.spinner("Leyendo texto…"):
                text, conf = ocr_extract(st.session_state.latest_image, tess_lang=tess_lang)
                st.session_state.ocr_text = text
                st.session_state.ocr_conf = conf
                st.toast("OCR completado", icon="✅")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.session_state.latest_image is not None:
            st.image(st.session_state.latest_image, channels="RGB", caption="Imagen utilizada", use_container_width=True)
    with c2:
        st.markdown(f"**Confianza media:** {st.session_state.ocr_conf:.1f}%")
        st.session_state.ocr_text = st.text_area("Texto detectado (editable)", value=st.session_state.ocr_text, height=260)
        colx, coly = st.columns([1, 1])
        with colx:
            st.download_button(
                "Descargar texto",
                data=st.session_state.ocr_text.encode("utf-8"),
                file_name="ocr.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with coly:
            if st.button("Limpiar", use_container_width=True):
                st.session_state.ocr_text = ""
                st.session_state.ocr_conf = 0.0

# ────────────────────────────── Traducción & TTS ────────────────────────────── #
with tab_tts:
    st.markdown("### Traducción y Audio")
    if not st.session_state.ocr_text:
        st.info("No hay texto para traducir. Ve a la pestaña **OCR** para generar texto o escribe abajo.")
    manual = st.text_area("Texto de entrada (opcional)", value=st.session_state.ocr_text, height=180)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        input_language = LANG_MAP[st.session_state.get('in_lang_label', 'Español')] if 'in_lang_label' in st.session_state else LANG_MAP['Español']
        output_language = LANG_MAP[st.session_state.get('out_lang_label', 'Inglés')] if 'out_lang_label' in st.session_state else LANG_MAP['Inglés']
    with col2:
        tld = TLD_MAP[st.session_state.get('tld_label', 'Default')] if 'tld_label' in st.session_state else TLD_MAP['Default']
    with col3:
        play_direct = st.toggle("Reproducir automáticamente", value=True)

    # (los selectboxes están en sidebar; aquí usamos sus valores de session_state)
    input_language = LANG_MAP.get(st.session_state.get('Lenguaje de entrada', 'Español'), 'es')
    output_language = LANG_MAP.get(st.session_state.get('Lenguaje de salida', 'Inglés'), 'en')
    tld = TLD_MAP.get(st.session_state.get('Acento de inglés (TLD)', 'Default'), 'com')
    show_output_text = st.session_state.get('Mostrar texto traducido', True)

    do_translate = st.button("Traducir y generar audio", use_container_width=True)
    if do_translate:
        with st.spinner("Traduciendo y sintetizando…"):
            try:
                audio_path, translated = text_to_speech(input_language, output_language, manual, tld)
                st.success("¡Listo!")
                if show_output_text:
                    st.markdown("**Texto traducido:**")
                    st.write(translated)
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/mp3", start_time=0, autoplay=play_direct)
                st.download_button("Descargar audio (MP3)", data=audio_bytes, file_name=os.path.basename(audio_path), mime="audio/mpeg")
            except Exception as e:
                st.error(f"Error durante la traducción o TTS: {e}")

# ────────────────────────────── Acerca de ────────────────────────────── #
with tab_about:
    st.markdown("""
**OCR + Traducción + TTS**

- Preprocesamiento (Grises, Umbral, Invertir, Desenfoque) para mejorar precisión.
- Soporta **Tesseract** en *spa/eng/eng+spa* y traducción con *googletrans*.
- Generación de voz con **gTTS** (acento por TLD) y descargas.
- Toggle **Claro/Oscuro** para legibilidad en cualquier display.

> Tip: para OCR en otros idiomas instala paquetes Tesseract (ej.: `tesseract-ocr-spa`).
""")
