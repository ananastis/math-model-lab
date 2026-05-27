import streamlit as st
import numpy as np
import plotly.graph_objects as go
import model  

# Налаштування сторінки
st.set_page_config(page_title="Лабораторна робота 5", layout="wide")

st.title("Моделювання динамічної системи: Хвильове рівняння (Група 5)")
st.markdown(
    "**Умови:** дискретні початкові та крайові умови, неперервні моделюючі функції $u_0, u_r$. Простір $R^2$, час $[0,T]$.")
st.markdown("---")

# Ініціалізація сховища стану (Session State)
if "sim_data" not in st.session_state:
    st.session_state.sim_data = None
    
# 1. ІНТЕРФЕЙС ВВЕДЕННЯ ПАРАМЕТРІВ 
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

st.sidebar.markdown("---")
# Момент часу для побудови графіка
t_plot = st.sidebar.slider("Часовий зріз для графіка (t)", 0.0, float(T), float(T) / 2)

# 2. КЕРУВАННЯ ТА ОБЧИСЛЕННЯ

if st.sidebar.button("Змоделювати систему", type="primary"):

    with st.spinner("Обчислення матриці Гріна та псевдообернення..."):
        try:
            # Формування спостережень та джерел
            obs, Y_obs = model.make_observations(func, c, x1a, x1b, x2a, x2b, T, R0, Rg)
            srcs = model.make_sources(x1a, x1b, x2a, x2b, M, c=c, T=T)

            # Побудова матриці та розв'язання
            A = model.build_matrix(obs, srcs, c)
            u, rank, residual = model.solve(A, Y_obs, lam=lam)

            # Зберігаємо важкі результати в сесію
            st.session_state.sim_data = {
                "srcs": srcs,
                "u": u,
                "rank": rank,
                "residual": residual,
                "matrix_shape": A.shape
            }
            st.success("Моделювання успішно завершено!")
        except Exception as e:
            st.error(f"Виникла помилка під час обчислень: {e}")
            st.session_state.sim_data = None

if st.session_state.sim_data is not None:
    data = st.session_state.sim_data
    
    # Підготовка сітки для 3D-графіка
    grid_size = 40  
    x1_grid = np.linspace(x1a, x1b, grid_size)
    x2_grid = np.linspace(x2a, x2b, grid_size)
    X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing='ij')

    # Швидкий векторизований розрахунок точного значення у
    Y_exact = func(X1, X2, t_plot, c)

    # Наближене значення y' (відновлене)
    Y_model = model.reconstruct(x1_grid, x2_grid, t_plot, data["srcs"], data["u"], c)

    # Оцінка точності
    norm_abs, norm_rel, max_err = model.accuracy(Y_exact, Y_model)

    # Виведення метрик
    st.subheader("Аналіз точності та стану системи")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ранг матриці A", f"{data['rank']} / {min(data['matrix_shape'])}")
    m2.metric("МНК-нев'язка", f"{data['residual']:.2e}")
    m3.metric("Відносна похибка", f"{norm_rel:.4f} %")
    m4.metric("Макс. похибка", f"{max_err:.4f}")

    st.markdown("---")
    st.subheader(f"Просторово-часовий стан при t = {t_plot:.2f}")

    # Масштаб осі Z будується ТІЛЬКИ на базі еталону. Це дійсно зріже числові викиди (шум) моделі.
    z_min = float(np.min(Y_exact) - 0.5)
    z_max = float(np.max(Y_exact) + 0.5)

    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        st.markdown("**Еталонний розв'язок (y)**")
        fig_exact = go.Figure(data=[go.Surface(z=Y_exact.T, x=x1_grid, y=x2_grid, colorscale='Viridis')])
        fig_exact.update_layout(
            scene=dict(zaxis=dict(range=[z_min, z_max])),
            margin=dict(l=0, r=0, b=0, t=0), height=500
        )
        st.plotly_chart(fig_exact, use_container_width=True)

    with col_plot2:
        st.markdown("**Змодельований стан (y')**")
        fig_model = go.Figure(data=[go.Surface(z=Y_model.T, x=x1_grid, y=x2_grid, colorscale='Plasma')])
        fig_model.update_layout(
            scene=dict(zaxis=dict(range=[z_min, z_max])), 
            margin=dict(l=0, r=0, b=0, t=0), 
            height=500
        )
        st.plotly_chart(fig_model, use_container_width=True)

    # Графік похибки
    st.markdown("---")
    st.subheader("Поле похибки (y - y')")
    diff = Y_exact - Y_model
    fig_err = go.Figure(data=[go.Surface(z=diff.T, x=x1_grid, y=x2_grid, colorscale='RdBu')])
    fig_err.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
    st.plotly_chart(fig_err, use_container_width=True)

else:
    st.info("Налаштуйте параметри зліва та натисніть **«Змоделювати систему»** для первинного розрахунку.")
