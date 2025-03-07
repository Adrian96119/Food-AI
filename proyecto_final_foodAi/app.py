import streamlit as st
import requests
import json
from PIL import Image

# URL de la API
API_URL = "https://fe4a-104-196-143-67.ngrok-free.app" 

st.set_page_config(page_title="Predicción de Platos", page_icon="🍽️", layout="wide")

st.title("🍽️ Predicción de Platos")

uploaded_file = st.file_uploader("📷 Sube una imagen del plato", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_container_width=True)

    with st.spinner("⏳ Analizando imagen..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{API_URL}/predict", files=files)

    if response.status_code == 200:
        result = response.json()
        st.success("✅ Predicción realizada con éxito")

        st.markdown("### 🍽️ Plato:")
        st.markdown(f"<p style='font-size: 22px; font-weight: bold;'>{result['plato']}</p>", unsafe_allow_html=True)

        st.markdown("### 🔥 Calorías:")
        st.markdown(f"<p style='font-size: 22px; font-weight: bold;'>{result['calorias']} kcal</p>", unsafe_allow_html=True)

        st.markdown("### 🛢️ Grasas:")
        st.markdown(f"<p style='font-size: 22px; font-weight: bold;'>{result['grasas']} g</p>", unsafe_allow_html=True)

        st.markdown("### 🍞 Carbohidratos:")
        st.markdown(f"<p style='font-size: 22px; font-weight: bold;'>{result['carbohidratos']} g</p>", unsafe_allow_html=True)

        st.markdown("### 💪 Proteínas:")
        st.markdown(f"<p style='font-size: 22px; font-weight: bold;'>{result['proteinas']} g</p>", unsafe_allow_html=True)

        # Mostrar ingredientes correctamente alineados
        if result.get("ingredientes") and result["ingredientes"] != "No disponible":
            st.markdown("### 📜 Ingredientes:")
            ingredientes_lista = result["ingredientes"].splitlines()
            ingredientes_limpios = [ing.strip("- ").strip() for ing in ingredientes_lista if ing.strip()]
            for ing in ingredientes_limpios:
                st.markdown(f"- {ing}")

        # Mostrar enlace de la receta si está disponible
        if result.get("url_receta") and result["url_receta"] != "No disponible":
            st.markdown(f"📌 **[Ver Receta]({result['url_receta']})**", unsafe_allow_html=True)

    else:
        st.error("❌ Error en la predicción")
