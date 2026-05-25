import streamlit as st
import numpy as np
import plotly.graph_objects as go
import model  # Імпортуємо твоє математичне ядро (файл model.py має бути в цій же папці)

# Налаштування сторінки
st.set_page_config(page_title="Лабораторна робота 5", layout="wide")

st.title("Моделювання динамічної системи: Хвильове рівняння (Група 5)")
st.markdown(
    "**Умови:** дискретні початкові та крайові умови, неперервні моделюючі функції $u_0, u_r$. Простір $R^2$, час $[0,T]$.")
st.markdown("---")

# ==========================================
# БЛОК 1: ІНТЕРФЕЙС ВВЕДЕННЯ ПАРАМЕТРІВ (Сайдбар)
# ==========================================
st.sidebar.header("1. Постановка задачі")

# Вибір тестової функції
func_names = list(model.TEST_FUNCTIONS.keys())
selected_func_name = st.sidebar.selectbox("Оберіть еталонну функцію y(x,t):", func_names)
func, func_formula = model.TEST_FUNCTIONS[selected_func_name]
st.sidebar.markdown(f"**Функція:** `{func_formula}`")

st.sidebar.header("2. Просторово-часова область")
col1, col2 = st.sidebar.columns(2)
x1a = col1.number_input("x1_a", value=-2.0)
x1b = col2.number_input("x1_b", value=2.0)
x2a = col1.number_input("x2_a", value=-2.0)
x2b = col2.number_input("x2_b", value=2.0)
T = st.sidebar.number_input("Час моделювання T", value=1.0, min_value=0.1)

st.sidebar.header("3. Параметри моделі")
c = st.sidebar.number_input("Швидкість хвилі (c)", value=1.0)
R0 = st.sidebar.slider("Початкові спостереження (R0)", 10, 200, 36)
Rg = st.sidebar.slider("Крайові спостереження (Rg)", 10, 200, 40)
M = st.sidebar.slider("К-ть джерел (M) для u0, ur", 10, 250, 64)
lam = st.sidebar.number_input("Регуляризація Тихонова (λ)", value=1e-6, format="%.1e")

# Момент часу для побудови графіка
t_plot = st.sidebar.slider("Часовий зріз для графіка (t)", 0.0, float(T), float(T) / 2)

# ==========================================
# БЛОК 2: КЕРУВАННЯ ТА ОБЧИСЛЕННЯ
# ==========================================
if st.sidebar.button("Змоделювати систему", type="primary"):

    with st.spinner("Обчислення матриці Гріна та псевдообернення..."):
        try:
            # 1. Формування спостережень та джерел
            obs, Y_obs = model.make_observations(func, c, x1a, x1b, x2a, x2b, T, R0, Rg)
            srcs = model.make_sources(x1a, x1b, x2a, x2b, M, c=c, T=T)

            # 2. Побудова матриці та розв'язання
            A = model.build_matrix(obs, srcs, c)
            u, rank, residual = model.solve(A, Y_obs, lam=lam)

            # 3. Підготовка сітки для 3D-графіка
            grid_size = 40  # Збільшено для кращої якості поверхні
            x1_grid = np.linspace(x1a, x1b, grid_size)
            x2_grid = np.linspace(x2a, x2b, grid_size)
            X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing='ij')

            # Точне значення y
            Y_exact = np.zeros_like(X1)
            for i in range(grid_size):
                for j in range(grid_size):
                    Y_exact[i, j] = func(X1[i, j], X2[i, j], t_plot, c)

            # Наближене значення y' (відновлене)
            Y_model = model.reconstruct(x1_grid, x2_grid, t_plot, srcs, u, c)

            # 4. Оцінка точності
            norm_abs, norm_rel, max_err = model.accuracy(Y_exact, Y_model)

            # ==========================================
            # БЛОК 3: ВИВЕДЕННЯ РЕЗУЛЬТАТІВ
            # ==========================================
            st.success("Моделювання успішно завершено!")

            # Метрики
            st.subheader("Аналіз точності та стану системи")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Ранг матриці A", f"{rank} / {min(A.shape)}")
            m2.metric("МНК-нев'язка", f"{residual:.2e}")
            m3.metric("Відносна похибка", f"{norm_rel:.4f} %")
            m4.metric("Макс. похибка", f"{max_err:.4f}")

            st.markdown("---")

            # 3D Графіки (ВИПРАВЛЕНО: додано .T для правильного відображення осей Plotly)
            st.subheader(f"Просторово-часовий стан при t = {t_plot:.2f}")

            z_min = min(np.min(Y_exact), np.min(Y_model)) - 0.5
            z_max = max(np.max(Y_exact), np.max(Y_model)) + 0.5

            col_plot1, col_plot2 = st.columns(2)

            with col_plot1:
                st.markdown("**Еталонний розв'язок (y)**")
                fig_exact = go.Figure(data=[go.Surface(z=Y_exact.T, x=x1_grid, y=x2_grid, colorscale='Viridis')])
                fig_exact.update_layout(scene=dict(zaxis=dict(range=[z_min, z_max])),
                                        margin=dict(l=0, r=0, b=0, t=0), height=500)
                st.plotly_chart(fig_exact, use_container_width=True)

            with col_plot2:
                st.markdown("**Змодельований стан (y')**")
                fig_model = go.Figure(data=[go.Surface(z=Y_model.T, x=x1_grid, y=x2_grid, colorscale='Plasma')])
                fig_model.update_layout(
                    scene=dict(
                        zaxis=dict(range=[z_min, z_max]) # Це "відріже" шум
                    ),
                    margin=dict(l=0, r=0, b=0, t=0), 
                    height=500
                )
                st.plotly_chart(fig_model, use_container_width=True)

            # Графік похибки
            st.markdown("---")
            st.subheader("Поле похибки (y - y')")
            diff = Y_exact - Y_model
            fig_err = go.Figure(data=[go.Surface(
                z=diff.T, x=x1_grid, y=x2_grid, colorscale='RdBu'
            )])
            fig_err.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
            st.plotly_chart(fig_err, use_container_width=True)
            

        except Exception as e:
            st.error(f"Виникла помилка під час обчислень: {e}")

else:
    st.info("👈 Налаштуйте параметри зліва та натисніть **«Змоделювати систему»**.")
