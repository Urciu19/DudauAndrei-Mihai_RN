import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from src.neural_network.infer import predict_image
from src.neural_network.postprocess import enrich_prediction


st.set_page_config(
    page_title="Recunoaștere Specii Vegetale",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 Sistem de recunoaștere a speciilor vegetale")
st.write("Încarcă o imagine cu o plantă/frunză și obții: specie, probabilitate, nume latin și descriere.")

# Sidebar settings
st.sidebar.header("Setări")
top_k = st.sidebar.slider("Top-K rezultate", min_value=1, max_value=5, value=3, step=1)

uploaded = st.file_uploader("Încarcă o imagine (.png / .jpg / .jpeg)", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Aștept o imagine...")
    st.stop()

# Afișează imaginea încărcată
img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Imagine încărcată", use_container_width=True)

# Salvează temporar pentru OpenCV (infer.py citește path)
suffix = Path(uploaded.name).suffix.lower()
if suffix not in [".png", ".jpg", ".jpeg"]:
    suffix = ".png"

with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp_path = tmp.name
    img.save(tmp_path)

# Predict
with st.spinner("Analizez imaginea..."):
    try:
        pred = predict_image(tmp_path, top_k=top_k)
        enriched = enrich_prediction(pred)
    except Exception as e:
        st.error(f"Eroare la predicție: {e}")
        st.stop()
    finally:
        # Curăță fișierul temporar
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

st.success("Gata!")

# Rezultat principal
st.subheader("Rezultat")
st.markdown(f"**Specie (label):** `{enriched['label']}`")
st.markdown(f"**Nume comun:** {enriched.get('common_name', 'N/A')}")
st.markdown(f"**Nume latin:** *{enriched.get('latin_name', 'N/A')}*")

conf = float(enriched.get("confidence", 0.0))
st.markdown(f"**Încredere:** {conf:.4f}")
st.progress(min(max(conf, 0.0), 1.0))

st.markdown("**Descriere:**")
st.write(enriched.get("description", "Descriere indisponibilă."))

# Top-K
st.subheader("Top-K predicții")
top_list = enriched.get("top_k", [])
if not top_list:
    st.write("N/A")
else:
    for i, item in enumerate(top_list, start=1):
        st.write(f"{i}. **{item['label']}** — {float(item['confidence']):.4f}")