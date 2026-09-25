import streamlit as st
import pandas as pd
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from PIL import Image
import pytesseract

# --- 1. Configuración de página ---
st.set_page_config(page_title="Herramientas NLP & Groq", layout="wide")
st.title("🛠️ Kit de NLP, OCR y Generación con Groq")

# --- Barra lateral para la API Key ---
st.sidebar.header("Configuración")
api_key = st.sidebar.text_input("Ingresa tu API Key de Groq", type="password")

# Cargar modelos disponibles si hay API Key
valid_models = []
if api_key:
    try:
        client = Groq(api_key=api_key)
        models_response = client.models.list()
        valid_models = [
            m.id for m in models_response.data 
            if "llama" not in m.id.lower() and "whisper" not in m.id.lower()
        ]
        if not valid_models:
            valid_models = [m.id for m in models_response.data if "whisper" not in m.id.lower()]
    except Exception as e:
        st.sidebar.error(f"Error con la API Key: {e}")
else:
    st.sidebar.warning("⚠️ Necesitas ingresar tu API Key de Groq para las funciones de generación.")

# --- Pestañas de la aplicación ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Tokenización", "Bag of Words", "Similitud Coseno", "Generación (Texto)", "OCR -> Generación"
])

# --- TAB 1: Tokenización ---
with tab1:
    st.header("Tokenización (Estilo GPT)")
    text_input = st.text_area("Texto a tokenizar:", "El procesamiento de lenguaje natural es fascinante.")
    encoding_name = st.selectbox("Selecciona el esquema", ["cl100k_base (GPT-4 / GPT-3.5)", "p50k_base (GPT-3)"])
    encoding_map = {"cl100k_base (GPT-4 / GPT-3.5)": "cl100k_base", "p50k_base (GPT-3)": "p50k_base"}
    
    if text_input:
        enc = tiktoken.get_encoding(encoding_map[encoding_name])
        tokens = enc.encode(text_input)
        
        st.subheader("Tokens obtenidos")
        colors = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#D0BAFF", "#FFBAF3"]
        html_content = "<div style='line-height: 2; font-size: 18px; display: flex; flex-wrap: wrap;'>"
        
        token_data = []
        for i, token_id in enumerate(tokens):
            token_str = enc.decode([token_id])
            color = colors[i % len(colors)]
            display_str = token_str.replace(" ", "␣").replace("\n", "↵")
            html_content += f"<div style='background-color: {color}; padding: 4px 8px; border-radius: 6px; margin: 4px; color: black; border: 1px solid #ddd;'><b>{display_str}</b> <span style='font-size: 12px; color: #555;'>({token_id})</span></div>"
            token_data.append({"Token (Texto)": display_str, "Token ID": token_id})
            
        html_content += "</div>"
        st.markdown(html_content, unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(token_data), use_container_width=True)

# --- TAB 2: Bag of Words ---
with tab2:
    st.header("Bag of Words (BoW)")
    bow_input = st.text_area(
        "Frases para el Corpus (una por línea):", 
        "Me gusta la inteligencia artificial\nLa inteligencia artificial es el futuro\nMe gusta aprender cosas nuevas"
    )
    if bow_input:
        corpus = [line.strip() for line in bow_input.split('\n') if line.strip()]
        if corpus:
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(corpus)
            bow_df = pd.DataFrame(
                X.toarray(), 
                columns=vectorizer.get_feature_names_out(), 
                index=[f"Frase {i+1}" for i in range(len(corpus))]
            )
            st.dataframe(bow_df, use_container_width=True)

# --- TAB 3: Similitud Coseno ---
with tab3:
    st.header("Similitud de Coseno")
    col1, col2 = st.columns(2)
    with col1:
        frase1 = st.text_area("Frase 1", "El gato come pescado felizmente")
    with col2:
        frase2 = st.text_area("Frase 2", "El felino se alimenta de peces")
        
    if st.button("Calcular Similitud", type="primary"):
        vectorizer_cos = CountVectorizer()
        try:
            X_cos = vectorizer_cos.fit_transform([frase1, frase2])
            sim = cosine_similarity(X_cos[0], X_cos[1])[0][0]
            st.success(f"### Distancia de Coseno (Similitud): {sim:.4f}")
        except ValueError:
            st.error("Por favor ingresa texto válido en ambas frases.")

# --- TAB 4: Generación Groq (Texto) ---
with tab4:
    st.header("Generación de Texto Clásica")
    if not api_key or not valid_models:
        st.info("Ingresa una API Key válida en la barra lateral para ver los modelos activos.")
    else:
        model_choice = st.selectbox("Modelos disponibles:", sorted(valid_models), key="model_tab4")
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            temperature = st.slider("Temperatura", 0.0, 2.0, 0.7, key="temp_tab4")
        with col_param2:
            max_tokens = st.slider("Max Tokens", 100, 4096, 1024, key="tokens_tab4")
            
        prompt = st.text_area("Escribe tu Prompt", "Explica brevemente la diferencia entre Bag of Words y Similitud Coseno.")
        
        if st.button("Generar Respuesta 🚀"):
            with st.spinner(f"Generando con {model_choice}..."):
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_choice,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    st.write(chat_completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error: {e}")

# --- TAB 5: OCR + Generación Groq ---
with tab5:
    st.header("Extracción de Texto (OCR) y Generación")
    st.markdown("Sube una imagen con texto. El sistema extraerá el contenido y podrás usarlo como base para un prompt con los modelos de Groq.")
    
    uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        
        with st.spinner("Extrayendo texto de la imagen..."):
            try:
                # lang='spa+eng' para reconocer español e inglés
                extracted_text = pytesseract.image_to_string(image, lang='spa+eng')
            except pytesseract.TesseractNotFoundError:
                extracted_text = ""
                st.error("❌ Tesseract no está instalado o no se encuentra en el PATH. Si estás en Streamlit Cloud, asegúrate de que el archivo 'packages.txt' esté en tu repositorio de GitHub.")
            except Exception as e:
                extracted_text = ""
                st.error(f"Error en OCR: {e}")
        
        if extracted_text.strip():
            st.success("Texto extraído con éxito.")
            # Permitimos que el usuario edite el texto por si el OCR tuvo algún fallo
            edited_text = st.text_area("Texto extraído (puedes editarlo):", value=extracted_text, height=150)
            
            st.markdown("### 🧠 Ampliar con IA")
            instruccion = st.text_input("¿Qué deseas hacer con este texto?", value="Resume este texto en tres puntos clave:")
            
            if not api_key or not valid_models:
                st.warning("⚠️ Ingresa tu API Key de Groq en la barra lateral para continuar.")
            else:
                model_choice_ocr = st.selectbox("Modelo para analizar OCR:", sorted(valid_models), key="model_tab5")
                col_param1_ocr, col_param2_ocr = st.columns(2)
                with col_param1_ocr:
                    temp_ocr = st.slider("Temperatura", 0.0, 2.0, 0.7, key="temp_tab5")
                with col_param2_ocr:
                    tokens_ocr = st.slider("Max Tokens", 100, 4096, 1024, key="tokens_tab5")
                
                final_prompt = f"{instruccion}\n\nTEXTO:\n{edited_text}"
                
                if st.button("Procesar Texto con Groq 🪄"):
                    with st.spinner(f"Analizando con {model_choice_ocr}..."):
                        try:
                            chat_completion = client.chat.completions.create(
                                messages=[{"role": "user", "content": final_prompt}],
                                model=model_choice_ocr,
                                temperature=temp_ocr,
                                max_tokens=tokens_ocr,
                            )
                            st.markdown("### Resultado:")
                            st.write(chat_completion.choices[0].message.content)
                        except Exception as e:
                            st.error(f"Error: {e}")
        elif uploaded_file is not None:
            st.warning("No se detectó texto en la imagen.")
