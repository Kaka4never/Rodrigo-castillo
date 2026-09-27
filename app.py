import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optimizador PLC Interactivo", layout="wide")

st.title("Aplicación Interactiva de Programación Lineal Continua (PLC)")
st.markdown("Experimenta cambiando las variables y restricciones para comprender la relación entre el modelo, la región factible y la solución óptima.")

# --- BARRA LATERAL: DEFINICIÓN DEL MODELO ---
st.sidebar.header("1. Definir Dimensiones")
tipo_opt = st.sidebar.radio("Objetivo:", ["Maximizar", "Minimizar"])
n_vars = st.sidebar.number_input("Número de variables", min_value=2, max_value=10, value=2)
n_cons = st.sidebar.number_input("Número de restricciones", min_value=1, max_value=15, value=3)

# --- CUERPO PRINCIPAL: ENTRADA DE DATOS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Función Objetivo")
    obj_coefs = []
    cols_obj = st.columns(n_vars)
    for i in range(n_vars):
        coef = cols_obj[i].number_input(f"Coef X{i+1}", value=1.0, key=f"obj_{i}")
        obj_coefs.append(coef)

with col2:
    st.subheader("Restricciones")
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    restricciones_raw = [] # Para calcular holguras luego
    
    for i in range(n_cons):
        cols_cons = st.columns(n_vars + 2)
        fila = []
        for j in range(n_vars):
            val = cols_cons[j].number_input(f"R{i+1}_X{j+1}", value=1.0, key=f"r_{i}_{j}")
            fila.append(val)
        
        tipo = cols_cons[n_vars].selectbox("", ["<=", ">=", "="], key=f"tipo_{i}")
        rhs = cols_cons[n_vars+1].number_input("Lado D.", value=10.0, key=f"rhs_{i}")
        
        restricciones_raw.append({"coefs": fila, "tipo": tipo, "rhs": rhs})
        
        if tipo == "<=":
            A_ub.append(fila)
            b_ub.append(rhs)
        elif tipo == ">=":
            A_ub.append([-x for x in fila])
            b_ub.append(-rhs)
        else:
            A_eq.append(fila)
            b_eq.append(rhs)

# Preparar datos para linprog
c = [-x for x in obj_coefs] if tipo_opt == "Maximizar" else obj_coefs
bounds = [(0, None) for _ in range(n_vars)] # Variables no negativas

if st.button("Resolver Modelo Óptimamente", type="primary"):
    res = linprog(c, 
                  A_ub=A_ub if A_ub else None, 
                  b_ub=b_ub if b_ub else None, 
                  A_eq=A_eq if A_eq else None, 
                  b_eq=b_eq if b_eq else None, 
                  bounds=bounds, 
                  method='highs')
    
    st.markdown("---")
    if res.success:
        st.success("¡Solución Óptima Encontrada!")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.subheader("Variables de Decisión")
            for i, val in enumerate(res.x):
                st.write(f"**X{i+1}** = {val:.4f}")
            
            z_val = -res.fun if tipo_opt == "Maximizar" else res.fun
            st.markdown(f"### Valor Objetivo (Z): {z_val:.4f}")
            
        with col_res2:
            st.subheader("Análisis de Restricciones (Saturación)")
            datos_rest = []
            for i, r in enumerate(restricciones_raw):
                valor_izq = sum(c * v for c, v in zip(r["coefs"], res.x))
                holgura = abs(r["rhs"] - valor_izq)
                estado = "Saturada (Activa)" if holgura < 1e-7 else "No Saturada"
                datos_rest.append([f"Restricción {i+1}", f"{valor_izq:.2f} {r['tipo']} {r['rhs']}", holgura, estado])
                
            df_rest = pd.DataFrame(datos_rest, columns=["Restricción", "Evaluación", "Holgura", "Estado"])
            st.dataframe(df_rest, hide_index=True)

        # GRAFICACIÓN 2D (Solo si son 2 variables)
        if n_vars == 2:
            st.markdown("---")
            st.subheader("Resolución Gráfica (2 Variables)")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Crear grid
            x = np.linspace(0, max(res.x[0]*2, 20), 400)
            y = np.linspace(0, max(res.x[1]*2, 20), 400)
            X, Y = np.meshgrid(x, y)
            
            # Capa base de región factible (True en todo el cuadrante positivo)
            region_factible = (X >= 0) & (Y >= 0)
            
            # Trazar restricciones y delimitar región factible
            for r in restricciones_raw:
                c1, c2 = r["coefs"][0], r["coefs"][1]
                rhs, tipo = r["rhs"], r["tipo"]
                
                # Para evitar divisiones por cero en el gráfico
                if c2 != 0:
                    y_line = (rhs - c1 * x) / c2
                    ax.plot(x, y_line, label=f"{c1}X1 + {c2}X2 {tipo} {rhs}")
                else:
                    ax.axvline(x=rhs/c1, label=f"{c1}X1 {tipo} {rhs}")
                
                # Lógica de sombreado
                if tipo == "<=":
                    region_factible = region_factible & (c1 * X + c2 * Y <= rhs)
                elif tipo == ">=":
                    region_factible = region_factible & (c1 * X + c2 * Y >= rhs)
                # (Para igualdades, la región es una línea, el sombreado 2D no aplica igual, se omite en máscara de áreas)

            # Sombrear la región factible
            ax.imshow(region_factible.astype(int), extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greens", alpha=0.3)
            
            # Marcar el punto óptimo
            ax.plot(res.x[0], res.x[1], 'ro', markersize=10, label="Punto Óptimo")
            ax.annotate(f"  ({res.x[0]:.2f}, {res.x[1]:.2f})", (res.x[0], res.x[1]), color='red', weight='bold')
            
            ax.set_xlim(0, max(res.x[0]*2, 10))
            ax.set_ylim(0, max(res.x[1]*2, 10))
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            st.pyplot(fig)

    else:
        st.error(f"El problema no tiene solución factible o es no acotado. Razón: {res.message}")