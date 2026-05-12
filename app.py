import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="US Vehicles Analysis", layout="wide")

st.title("US Vehicles Analysis - EDA")
st.markdown("Análisis exploratorio interactivo de vehículos en Estados Unidos")

try:
    # Cargar datos
    @st.cache_data
    def load_data():
        return pd.read_csv("data/raw/vehicles_us.csv")

    df = load_data()

    # Configurar matplotlib
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette('husl')

    # ===== METRICAS PRINCIPALES =====
    st.header("Métricas Principales")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de vehículos", f"{len(df):,}")
    with col2:
        st.metric("Precio promedio", f"${df['price'].mean():,.0f}")
    with col3:
        st.metric("Año promedio", f"{df['model_year'].mean():.0f}")
    with col4:
        st.metric("Cilindros promedio", f"{df['cylinders'].mean():.1f}")

    st.divider()

    # ===== HISTOGRAMAS =====
    st.header("Histogramas")
    
    hist_col = st.selectbox(
        "Selecciona una variable para el histograma:",
        ["price", "model_year", "cylinders", "days_listed", "odometer"]
    )

    col_left, col_center, col_right = st.columns([0.5, 1.5, 0.5])
    
    with col_center:
        if hist_col == "price":
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(df['price'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
            ax.set_title('Distribución de Precios de Vehículos', fontsize=14, fontweight='bold')
            ax.set_xlabel('Precio ($)', fontsize=12)
            ax.set_ylabel('Frecuencia', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig, use_container_width=False)
            
        elif hist_col == "model_year":
            fig, ax = plt.subplots(figsize=(5, 3))
            df_year = df.dropna(subset=['model_year'])
            ax.hist(df_year['model_year'], bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
            ax.set_title('Distribución de Años de Modelos', fontsize=14, fontweight='bold')
            ax.set_xlabel('Año del Modelo', fontsize=12)
            ax.set_ylabel('Frecuencia', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig, use_container_width=False)
            
        elif hist_col == "cylinders":
            fig, ax = plt.subplots(figsize=(5, 3))
            df_cyl = df.dropna(subset=['cylinders'])
            ax.hist(df_cyl['cylinders'], bins=10, color='#2ecc71', edgecolor='black', alpha=0.7)
            ax.set_title('Distribución de Cilindros', fontsize=14, fontweight='bold')
            ax.set_xlabel('Número de Cilindros', fontsize=12)
            ax.set_ylabel('Frecuencia', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig, use_container_width=False)
            
        elif hist_col == "days_listed":
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(df['days_listed'], bins=30, color='#f39c12', edgecolor='black', alpha=0.7)
            ax.set_title('Distribución de Días Listado', fontsize=14, fontweight='bold')
            ax.set_xlabel('Días Listado', fontsize=12)
            ax.set_ylabel('Frecuencia', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig, use_container_width=False)
            
        else:  # odometer
            fig, ax = plt.subplots(figsize=(5, 3))
            df_odo = df.dropna(subset=['odometer'])
            ax.hist(df_odo['odometer'], bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
            ax.set_title('Distribución de Odómetro', fontsize=14, fontweight='bold')
            ax.set_xlabel('Odómetro (millas)', fontsize=12)
            ax.set_ylabel('Frecuencia', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig, use_container_width=False)

    st.divider()

    # ===== SCATTER PLOTS =====
    st.header("Gráficos de Dispersión")
    
    scatter_type = st.selectbox(
        "Selecciona un scatter plot:",
        ["Año vs Precio", "Cilindros vs Precio", "Días Listado vs Precio", "Odómetro vs Precio"]
    )

    col_left, col_center, col_right = st.columns([0.5, 1.5, 0.5])
    
    with col_center:
        if scatter_type == "Año vs Precio":
            df_clean = df.dropna(subset=['model_year', 'price'])
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.scatter(df_clean['model_year'], df_clean['price'], alpha=0.5, s=20, color='#3498db')
            ax.set_title('Año del Modelo vs Precio', fontsize=14, fontweight='bold')
            ax.set_xlabel('Año del Modelo', fontsize=12)
            ax.set_ylabel('Precio ($)', fontsize=12)
            ax.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=False)
            
        elif scatter_type == "Cilindros vs Precio":
            df_scatter = df.dropna(subset=['cylinders', 'price', 'type'])
            fig, ax = plt.subplots(figsize=(5, 3))
            for vtype in df_scatter['type'].unique():
                data = df_scatter[df_scatter['type'] == vtype]
                ax.scatter(data['cylinders'], data['price'], label=vtype, alpha=0.6, s=30)
            ax.set_title('Cilindros vs Precio (coloreado por tipo)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Número de Cilindros', fontsize=12)
            ax.set_ylabel('Precio ($)', fontsize=12)
            ax.legend(loc='best', fontsize=10)
            ax.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=False)
            
        elif scatter_type == "Días Listado vs Precio":
            df_scatter2 = df.dropna(subset=['days_listed', 'price'])
            fig, ax = plt.subplots(figsize=(5, 3))
            for cond in df_scatter2['condition'].unique():
                data = df_scatter2[df_scatter2['condition'] == cond]
                ax.scatter(data['days_listed'], data['price'], label=cond, alpha=0.5, s=20)
            ax.set_title('Días Listado vs Precio (coloreado por condición)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Días Listado', fontsize=12)
            ax.set_ylabel('Precio ($)', fontsize=12)
            ax.legend(loc='best', fontsize=10)
            ax.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=False)
            
        else:  # Odómetro vs Precio
            df_scatter3 = df.dropna(subset=['odometer', 'price'])
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.scatter(df_scatter3['odometer'], df_scatter3['price'], alpha=0.5, s=20, color='#e74c3c')
            ax.set_title('Odómetro vs Precio', fontsize=14, fontweight='bold')
            ax.set_xlabel('Odómetro (millas)', fontsize=12)
            ax.set_ylabel('Precio ($)', fontsize=12)
            ax.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=False)

    st.divider()

    # ===== GRÁFICOS ADICIONALES =====
    st.header("Análisis Adicionales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución por Tipo de Vehículo")
        vehicle_counts = df['type'].value_counts()
        
        # Agrupar valores menores al 5% en "Otros"
        total = vehicle_counts.sum()
        threshold = total * 0.05
        others_count = vehicle_counts[vehicle_counts < threshold].sum()
        vehicle_counts_filtered = vehicle_counts[vehicle_counts >= threshold]
        
        if others_count > 0:
            vehicle_counts_filtered['Otros'] = others_count
        
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.pie(vehicle_counts_filtered.values, labels=vehicle_counts_filtered.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('Tipos de Vehículos', fontsize=11, fontweight='bold')
        st.pyplot(fig, use_container_width=False)

    with col2:
        st.subheader("Precios por Tipo (Box Plot)")
        fig, ax = plt.subplots(figsize=(4.2, 3))
        df.boxplot(column='price', by='type', ax=ax)
        ax.set_title('Distribución de Precios por Tipo', fontsize=11, fontweight='bold')
        ax.set_xlabel('Tipo de Vehículo', fontsize=9)
        ax.set_ylabel('Precio ($)', fontsize=9)
        ax.tick_params(axis='x', rotation=45)
        plt.suptitle('')
        st.pyplot(fig, use_container_width=False)

    st.divider()

    # ===== MATRIZ DE CORRELACION =====
    st.header("Matriz de Correlación")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    col_left, col_center, col_right = st.columns([0.3, 1.4, 0.3])
    with col_center:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, annot_kws={"size": 8}, ax=ax)
        ax.set_title('Matriz de Correlación', fontsize=12, fontweight='bold')
        st.pyplot(fig, use_container_width=False)

    st.divider()

    # ===== INSIGHTS =====
    st.header("Insights Clave")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Precios")
        st.write(f"**Mínimo:** ${df['price'].min():,.0f}")
        st.write(f"**Máximo:** ${df['price'].max():,.0f}")
        st.write(f"**Promedio:** ${df['price'].mean():,.0f}")
        st.write(f"**Mediana:** ${df['price'].median():,.0f}")
    
    with col2:
        st.subheader("Modelos")
        st.write(f"**Año mínimo:** {df['model_year'].min():.0f}")
        st.write(f"**Año máximo:** {df['model_year'].max():.0f}")
        st.write(f"**Año promedio:** {df['model_year'].mean():.1f}")
    
    with col3:
        st.subheader("Otros")
        top_type = df['type'].value_counts().index[0]
        top_count = df['type'].value_counts().iloc[0]
        st.write(f"**Vehículo más común:** {top_type}")
        st.write(f"**Cilindros promedio:** {df['cylinders'].mean():.1f}")
        st.write(f"**Días listado promedio:** {df['days_listed'].mean():.1f}")

    st.divider()

    # ===== DATOS COMPLETOS =====
    st.header("Datos Completos")
    st.dataframe(df, use_container_width=True)

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.info("Verifica que el archivo `data/raw/vehicles_us.csv` exista.")