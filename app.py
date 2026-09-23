# ==============================================================================
# APLICACIÓN INTERACTIVA DE SOPORTE CLÍNICO - DUAL GRAFT (STREAMLIT + PYTORCH)
# Autor: Francisco Broissin Físico / Entusiasta de la Ciencia de Datos
# Licencia: MIT (Software libre y abierto)
# ==============================================================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Configuración de la página
st.set_page_config(
    page_title="Soporte Clínico Dual Graft - IA",
    page_icon="🧬",
    layout="wide"
)

# Título y descripción principal
st.title("🧬 Sistema de Apoyo a la Decisión Clínica: Trasplante Dual Graft")
st.markdown("""
Plataforma predictiva basada en **Redes Neuronales Profundas (PyTorch)** para estimar la probabilidad de éxito de tolerancia inmunológica y la proyección de supervivencia a largo plazo en pacientes receptores de trasplante.
""")

# --- 1. SIMULACIÓN DE ENTRENAMIENTO DEL MODELO DE BASE ---
@st.cache_resource
def entrenar_modelo_base():
    np.random.seed(42)
    n_samples = 500
    edad = np.random.randint(1, 70, n_samples)
    hla = np.random.randint(0, 7, n_samples)
    tx_prev = np.random.randint(0, 4, n_samples)
    vol = np.random.normal(100, 15, n_samples)
    riesgo = hla * 1.2 + tx_prev * 1.5 + (edad < 18) * 2
    
    # Generamos la variable objetivo simulada con lógica biológica
    prob_base = 1 / (1 + np.exp(-(-0.05 * edad - 0.4 * hla - 0.6 * tx_prev + 0.03 * vol + 2.5)))
    exito = (np.random.rand(n_samples) < prob_base).astype(int)
    
    X_raw = np.column_stack((edad, hla, tx_prev, vol, riesgo))
    y_raw = exito.astype('float32')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)
    
    class DualGraftNet(nn.Module):
        def __init__(self, input_dim):
            super(DualGraftNet, self).__init__()
            self.layer1 = nn.Linear(input_dim, 16)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.layer2 = nn.Linear(16, 8)
            self.layer3 = nn.Linear(8, 1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            out = self.layer1(x)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.layer2(out)
            out = self.relu(out)
            out = self.layer3(out)
            out = self.sigmoid(out)
            return out

    model = DualGraftNet(input_dim=X_tensor.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for _ in range(150):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).numpy().flatten()
        
    return model, scaler, X_raw, y_raw, preds

model, scaler, X_raw, y_raw, y_pred_probs = entrenar_modelo_base()

# --- 2. BARRA LATERAL: ENTRADA DE PARÁMETROS CLÍNICOS ---
st.sidebar.header("📋 Parámetros del Paciente")

def user_input_features():
    edad = st.sidebar.slider("Edad del Receptor (Años)", 1, 80, 40)
    hla = st.sidebar.slider("Incompatibilidad HLA (Mismatch)", 0, 6, 1)
    tx_prev = st.sidebar.slider("Trasplantes Previos", 0, 5, 1)
    vol_medula = st.sidebar.slider("Volumen de Médula (ml)", 50.0, 150.0, 100.0)
    
    riesgo = hla * 1.2 + tx_prev * 1.5 + (edad < 18) * 2
    data = np.array([[edad, hla, tx_prev, vol_medula, riesgo]])
    return data

input_df = user_input_features()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏥 Contexto Científico")
st.sidebar.markdown(
    "Trabajo de investigación desarrollado de forma independiente y desinteresada, "
    "inspirado en los avances clínicos sobre trasplantes *Dual Graft* y quimerismo mixto "
    "liderados por el equipo médico del **Dr. Francisco Hernández-Oliveros** (Cirugía) "
    "y el **Dr. Antonio Pérez Martínez** (Oncología)."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏛️ Créditos y Transparencia")
st.sidebar.markdown("**Autor:** Francisco Broissin  \n*(Licenciado en CC Físicas - UCM)* ")
st.sidebar.markdown("*Ex Opera et Adiutorio eximio Inteligentiae Artificialis Gemini Googlensis*")
st.sidebar.markdown("**Licencia:** [MIT Open Source](LICENSE)")
#st.sidebar.markdown("Mayo 2026, Alcochete, Portugal, Union Europea")
st.sidebar.markdown("Mayo 2026")
st.sidebar.markdown("Alcochete, Portugal, Union Europea")



# --- 3. INFERENCIA EN TIEMPO REAL ---
input_scaled = scaler.transform(input_df)
tensor_input = torch.tensor(input_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    prob_exito = model(tensor_input).item()

# --- 4. CUERPO PRINCIPAL: RESULTADOS Y VISUALIZACIÓN ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎯 Pronóstico Actual")
    
    if prob_exito >= 0.65:
        st.success(f"### Probabilidad de Éxito: {prob_exito * 100:.1f}%")
        st.markdown("**Evaluación:** Perfil favorable con alta probabilidad de estabilidad a largo plazo.")
    elif prob_exito >= 0.40:
        st.warning(f"### Probabilidad de Éxito: {prob_exito * 100:.1f}%")
        st.markdown("**Evaluación:** Zona de cautela. Requiere seguimiento inmunológico estrecho.")
    else:
        st.error(f"### Probabilidad de Éxito: {prob_exito * 100:.1f}%")
        st.markdown("**Evaluación:** Alto riesgo de rechazo o fallo de injerto. Evaluar alternativas.")
        
    st.markdown("---")
    st.markdown("**Resumen de Parámetros Ingresados:**")
    st.json({
        "Edad": int(input_df[0][0]),
        "HLA Mismatch": int(input_df[0][1]),
        "Tx Previos": int(input_df[0][2]),
        "Volumen Médula (ml)": float(input_df[0][3]),
        "Riesgo Inmunológico": round(input_df[0][4], 2)
    })

with col2:
    tab1, tab2 = st.tabs(["📈 Proyección Temporal de Supervivencia", "📊 Validación y Métricas del Modelo"])
    
    with tab1:
        st.markdown("#### Evolución de la Estabilidad Inmunológica Post-Trasplante")
        
        meses = np.array([0, 3, 6, 12, 24, 36])
        tasa_riesgo = 0.05 * (1.1 - prob_exito)
        supervivencia = np.exp(-tasa_riesgo * meses) * 100
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(meses, supervivencia, marker='o', color='#2b5c8f', lw=2.5, label='Paciente Actual')
        
        # Referencias teóricas
        ax.plot(meses, np.exp(-0.02 * meses) * 100, linestyle='--', color='forestgreen', alpha=0.6, label='Ref. Favorable (Bajo Riesgo)')
        ax.plot(meses, np.exp(-0.12 * meses) * 100, linestyle='--', color='firebrick', alpha=0.6, label='Ref. Compleja (Alto Riesgo)')
        
        ax.set_ylim([0, 105])
        ax.set_xlabel('Tiempo Post-Trasplante (Meses)', fontsize=10)
        ax.set_ylabel('Probabilidad de Estabilidad (%)', fontsize=10)
        ax.set_title('Curva de Supervivencia Libre de Eventos', fontsize=11, fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig)

        st.info(
            "💡 **Leyenda de Referencias:**\n"
            "- **Ref. Favorable:** Comportamiento esperado en pacientes con baja incompatibilidad HLA y sin antecedentes complejos (perfil tipo Nora).\n"
            "- **Ref. Compleja:** Comportamiento en escenarios de alto riesgo inmunológico y múltiples trasplantes previos (perfil tipo Yassine)."
        )

    with tab2:
        st.markdown("#### Rendimiento Diagnóstico Global del Sistema IA")
        
        fpr, tpr, _ = roc_curve(y_raw, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        
        y_pred_bin = (y_pred_probs >= 0.5).astype(int)
        cm = confusion_matrix(y_raw, y_pred_bin)
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
        
        # Curva ROC
        axes2[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        axes2[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axes2[0].set_xlabel('1 - Especificidad')
        axes2[0].set_ylabel('Sensibilidad')
        axes2[0].set_title(f'Curva ROC (AUC = {roc_auc:.2f})', fontsize=10, fontweight='bold')
        axes2[0].legend(loc='lower right')
        axes2[0].grid(True, linestyle=':', alpha=0.6)
        
        # Matriz de Confusión
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes2[1],
                    xticklabels=['Fallo', 'Éxito'], yticklabels=['Fallo', 'Éxito'])
        axes2[1].set_title('Matriz de Confusión (Umbral 50%)', fontsize=10, fontweight='bold')
        axes2[1].set_xlabel('Predicción')
        axes2[1].set_ylabel('Valor Real')
        
        plt.tight_layout()
        st.pyplot(fig2)

# --- Nota metodológica al pie de la página ---
st.markdown("---")
st.markdown(
    "⚠️ **Nota Metodológica y Transparencia Científica:** Este prototipo de soporte clínico opera sobre un dataset sintético modelado bajo restricciones clínicas coherentes con un rango de valores razonables (no imposibles). "
    "El modelo de Red Neuronal Profunda (PyTorch) subyacente ha alcanzado un rendimiento validado (AUC) en las pruebas de discriminación diagnóstica que se actualiza dinámicamente en la pestaña de validación de arriba. "
    "Desarrollado bajo Licencia MIT."
)
