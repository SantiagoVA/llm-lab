import streamlit as st
import pandas as pd
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# --- 1. Configuración de página ---
st.set_page_config(page_title="Herramientas NLP & Groq", layout="wide")
st.title("🛠️ Kit de NLP y Generación con Groq")

# --- Barra lateral para la API Key ---
st.sidebar.header("Configuración")
api_key = st.sidebar.text_input("Ingresa tu API Key de Groq", type="password")
if not api_key:
    st.sidebar.warning("⚠️ Necesitas ingresar tu API Key de Groq para usar la sección de generación.")

# --- Pestañas de la aplicación ---
tab1, tab2, tab3, tab4 = st.tabs(["Tokenización", "Bag of Words", "Similitud Coseno", "Generación (Groq)"])

# --- TAB 1: Tokenización ---
with tab1:
    st.header("Tokenización (Estilo GPT)")
    st.markdown("Explora cómo se divide el texto. Usamos la librería `tiktoken` que contiene los esquemas de tokenización de los modelos GPT de OpenAI.")
    
    text_input = st.text_area("Texto a tokenizar:", "El procesamiento de lenguaje natural es fascinante.")
    encoding_name = st.selectbox(
        "Selecciona el esquema de tokenización", 
        ["cl100k_base (GPT-4 / GPT-3.5)", "p50k_base (GPT-3)"]
    )
    
    encoding_map = {
        "cl100k_base (GPT-4 / GPT-3.5)": "cl100k_base", 
        "p50k_base (GPT-3)": "p50k_base"
    }
    
    if text_input:
        enc = tiktoken.get_encoding(encoding_map[encoding_name])
        tokens = enc.encode(text_input)
        
        st.subheader("Tokens obtenidos (visualización con colores)")
        colors = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#D0BAFF", "#FFBAF3"]
        html_content = "<div style='line-height: 2; font-size: 18px; display: flex; flex-wrap: wrap;'>"
        
        token_data = []
        for i, token_id in enumerate(tokens):
            token_str = enc.decode([token_id])
            color = colors[i % len(colors)]
            display_str = token_str.replace(" ", "␣").replace("\n", "↵")
            
            # HTML para el token coloreado
            html_content += f"<div style='background-color: {color}; padding: 4px 8px; border-radius: 6px; margin: 4px; color: black; border: 1px solid #ddd;'><b>{display_str}</b> <span style='font-size: 12px; color: #555;'>({token_id})</span></div>"
            
            token_data.append({"Token (Texto)": display_str, "Token ID": token_id})
            
        html_content += "</div>"
        st.markdown(html_content, unsafe_allow_html=True)
        
        st.subheader("Tabla de Tokens e IDs")
        st.dataframe(pd.DataFrame(token_data), use_container_width=True)

# --- TAB 2: Bag of Words ---
with tab2:
    st.header("Bag of Words (BoW)")
    st.markdown("Ingresa varias frases (una por línea) para generar la matriz de recuento de palabras.")
    
    bow_input = st.text_area(
        "Frases para el Corpus:", 
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
    st.markdown("Mide qué tan similares son dos frases calculando el coseno del ángulo entre sus vectores (usando conteo de palabras).")
    
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
            if sim == 0:
                st.info("Un valor de 0.0000 significa que las frases no comparten ninguna palabra exacta.")
        except ValueError:
            st.error("Por favor ingresa texto válido en ambas frases.")

# --- TAB 4: Generación Groq ---
with tab4:
    st.header("Generación de Texto (API Groq)")
    st.markdown("Configura los parámetros y utiliza los modelos disponibles en Groq. *Nota: Hemos excluido LLaMA y los modelos deprecados.*")
    
    # Catálogo actualizado de modelos NO LLAMA en Groq
    model_choice = st.selectbox(
        "Catálogo de Modelos (Sin LLaMA)", 
        ["gemma2-9b-it", "gemma-7b-it"]
    )
    
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        temperature = st.slider("Temperatura (Creatividad)", 0.0, 2.0, 0.7)
    with col_param2:
        max_tokens = st.slider("Max Tokens (Longitud máxima)", 100, 4096, 1024)
        
    prompt = st.text_area("Escribe tu Prompt", "Explica brevemente la diferencia entre Bag of Words y Similitud Coseno.")
    
    if st.button("Generar Respuesta 🚀"):
        if not api_key:
            st.error("❌ Por favor, ingresa tu API Key de Groq en la barra lateral.")
        else:
            try:
                client = Groq(api_key=api_key)
                with st.spinner(f"Generando respuesta usando {model_choice}..."):
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_choice,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    st.write(chat_completion.choices[0].message.content)
            except Exception as e:
                st.error(f"Error en la comunicación con Groq: {e}")
