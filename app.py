import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="SECOP II - Dashboard", layout="wide")

kpis = pd.read_csv("kpis.csv").iloc[0]
gasto_dep = pd.read_csv("gasto_departamento.csv")
predicciones = pd.read_csv("predicciones_muestra.csv")
cm = pd.read_csv("matriz_confusion.csv", index_col=0)
importancias = pd.read_csv("importancias.csv")
modelo = joblib.load("modelo_adjudicacion.joblib")

st.title("SECOP II - Contratacion publica en Colombia")
st.caption("ACA 2 - Herramientas de Big Data - Prediccion de adjudicacion de contratos")

# ---- KPIs ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de procesos", f"{int(kpis['total_procesos']):,}")
c2.metric("% Adjudicados", f"{kpis['pct_adjudicados']:.1f}%")
c3.metric("Gasto total", f"${kpis['gasto_total']:,.0f}")
c4.metric("Precio promedio", f"${kpis['precio_promedio']:,.0f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Accuracy del modelo", f"{kpis['accuracy']:.2%}")
c6.metric("Precision", f"{kpis['precision']:.2%}")
c7.metric("Recall", f"{kpis['recall']:.2%}")
c8.metric("F1-score", f"{kpis['f1']:.2%}")

st.divider()

col_izq, col_der = st.columns(2)

with col_izq:
    st.subheader("Gasto por departamento")
    top_n = st.slider("Cuantos departamentos mostrar", 5, 15, 10)
    fig1 = px.bar(gasto_dep.head(top_n), x="departamento_entidad", y="gasto_total",
                  color="pct_adjudicados", title="Gasto total por departamento")
    st.plotly_chart(fig1, use_container_width=True)

with col_der:
    st.subheader("Matriz de confusion del modelo")
    fig2 = px.imshow(cm, text_auto=True, color_continuous_scale="Greens",
                      labels=dict(x="Prediccion", y="Real"))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

col_izq2, col_der2 = st.columns(2)

with col_izq2:
    st.subheader("Variables que mas influyen en la adjudicacion")
    top_imp = pd.concat([importancias.head(8), importancias.tail(8)])
    fig3 = px.bar(top_imp, x="coeficiente", y="variable", orientation="h",
                  title="Coeficientes del modelo (positivo = mas probabilidad de adjudicacion)")
    st.plotly_chart(fig3, use_container_width=True)

with col_der2:
    st.subheader("Distribucion de probabilidad predicha")
    fig4 = px.histogram(predicciones, x="probabilidad", color="real", nbins=30,
                         title="Probabilidad predicha de adjudicacion")
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

st.subheader("Probar el modelo con datos nuevos")
col1, col2, col3 = st.columns(3)
precio_base = col1.number_input("Precio base", min_value=0, value=50000000, step=1000000)
duracion = col2.number_input("Duracion (dias)", min_value=0, value=30)
proveedores_invitados = col3.number_input("Proveedores invitados", min_value=0, value=3)

col4, col5 = st.columns(2)
respuestas_al_procedimiento = col4.number_input("Respuestas al procedimiento", min_value=0, value=2)
modalidad_de_contratacion = col5.selectbox("Modalidad de contratacion",
    sorted(importancias["variable"].str.extract(r"modalidad_de_contratacion_(.*)").dropna()[0].unique()))
departamento_entidad = st.selectbox("Departamento",
    sorted(gasto_dep["departamento_entidad"].unique()))

if st.button("Predecir"):
    entrada = pd.DataFrame([{
        "precio_base": precio_base,
        "duracion": duracion,
        "proveedores_invitados": proveedores_invitados,
        "respuestas_al_procedimiento": respuestas_al_procedimiento,
        "modalidad_de_contratacion": modalidad_de_contratacion,
        "departamento_entidad": departamento_entidad,
    }])
    prob = modelo.predict_proba(entrada)[0, 1]
    st.success(f"Probabilidad de adjudicacion: {prob:.1%}")
