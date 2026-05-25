"""
model.py — Математичне ядро лабораторної роботи 5
Хвильове рівняння в R², простір [0,T]
L(∂s) = ∂²t − c²(∂²x1 + ∂²x2)
G(s)  = H(t − r/c) / (2πc√(c²t² − r²))
"""

import numpy as np


# ──────────────────────────────────────────────
# 1. ТЕСТОВІ ФУНКЦІЇ y(x1, x2, t)
# ──────────────────────────────────────────────

def y_sin_cos(x1, x2, t, c):
    """sin(x1)·cos(x2)·cos(c·t)  — точно задовольняє L(y)=0"""
    return np.sin(x1) * np.cos(x2) * np.cos(c * t)

def y_gauss(x1, x2, t, c):
    """exp(-(x1²+x2²))·cos(c·t)"""
    return np.exp(-(x1**2 + x2**2)) * np.cos(c * t)

def y_wave(x1, x2, t, c):
    """cos(x1+x2)·cos(c·t·√2)  — точно задовольняє L(y)=0"""
    return np.cos(x1 + x2) * np.cos(c * t * np.sqrt(2))

def y_poly(x1, x2, t, c):
    """x1²·x2·cos(t)"""
    return x1**2 * x2 * np.cos(t)

TEST_FUNCTIONS = {
    "sin_cos": (y_sin_cos, "sin(x1)·cos(x2)·cos(c·t)"),
    "gauss":   (y_gauss,   "exp(-(x1²+x2²))·cos(c·t)"),
    "wave":    (y_wave,    "cos(x1+x2)·cos(c·t·√2)"),
    "poly":    (y_poly,    "x1²·x2·cos(t)"),
}


# ──────────────────────────────────────────────
# 2. ФУНКЦІЯ ГРІНА  G(s, s')
# ──────────────────────────────────────────────

def green(x1, x2, t, x1s, x2s, ts, c):
    """
    Функція Гріна хвильового рівняння в R²:
      G(s, s') = H(t-t' - r/c) / (2πc·√(c²(t-t')² - r²))
    де r = √((x1-x1')² + (x2-x2')²)

    Повертає 0 якщо поза світловим конусом.
    """
    x1  = np.asarray(x1,  dtype=float)
    x2  = np.asarray(x2,  dtype=float)
    t   = np.asarray(t,   dtype=float)
    dt   = t - float(ts)
    r2   = (x1 - float(x1s))**2 + (x2 - float(x2s))**2
    disc = c**2 * dt**2 - r2

    if x1.ndim == 0:               # scalar call
        if float(dt) <= 1e-12 or float(disc) <= 1e-12:
            return 0.0
        return float(1.0 / (2.0 * np.pi * c * np.sqrt(float(disc))))
    else:                          # array call
        mask = (dt > 1e-12) & (disc > 1e-12)
        out  = np.zeros(x1.shape, dtype=float)
        out[mask] = 1.0 / (2.0 * np.pi * c * np.sqrt(disc[mask]))
        return out


# ──────────────────────────────────────────────
# 3. ТОЧКИ СПОСТЕРЕЖЕНЬ  (початкові + крайові)
# ──────────────────────────────────────────────

def make_observations(fn, c, x1a, x1b, x2a, x2b, T, R0, Rg):
    """
    Формує дискретні спостереження з 'справжньої' функції y.

    Початкові (t=0): рівномірна сітка √R0 × √R0 по (x1, x2)
    Крайові  (x1=x1a або x1b): рівномірно по t

    Повертає:
      obs  — список dict з ключами: type, x1, x2, t, Y
      Y    — numpy вектор значень спостережень
    """
    obs = []

    # — початкові (t = 0) —
    n = max(2, int(np.ceil(np.sqrt(R0))))
    x1_grid = np.linspace(x1a, x1b, n)
    x2_grid = np.linspace(x2a, x2b, n)
    for x1 in x1_grid:
        for x2 in x2_grid:
            if len([o for o in obs if o["type"]=="init"]) >= R0:
                break
            val = fn(x1, x2, 0.0, c)
            obs.append({"type":"init", "x1":x1, "x2":x2, "t":0.0, "Y":val})

    # — крайові (x1 = x1a або x1b) —
    for i in range(Rg):
        t_  = T * i / max(Rg - 1, 1)
        x1_ = x1a if i % 2 == 0 else x1b
        x2_ = x2a + (x2b - x2a) * (i / max(Rg - 1, 1))
        val = fn(x1_, x2_, t_, c)
        obs.append({"type":"bound", "x1":x1_, "x2":x2_, "t":t_, "Y":val})

    Y = np.array([o["Y"] for o in obs])
    return obs, Y


# ──────────────────────────────────────────────
# 4. ТОЧКИ ДЖЕРЕЛ  s'm  (зовнішня область t < 0)
# ──────────────────────────────────────────────

def make_sources(x1a, x1b, x2a, x2b, M, T_ext=5.0):
    """
    Розміщує M точок джерел у зовнішній ЧАСОВІЙ області (t = -T_ext).

    Відповідно до п.2.2 методики Стояна:
      – моделюючі фактори u⁰(s) визначені при t < 0 (для початкових умов)
      – джерела розміщені рівномірно всередині просторової області
      – T_ext достатньо велике щоб конус G охоплював усі точки спостережень:
          c · T_ext > r_max = √2 · max(|x1|, |x2|)

    За умов x1,x2 ∈ [-2,2], c=1: T_ext ≥ √2·2 ≈ 2.83  →  T_ext=3.0
    """
    n    = max(2, int(np.ceil(np.sqrt(M))))
    srcs = []
    dx1  = (x1b - x1a) * 0.3          # невеликий відступ за межі
    dx2  = (x2b - x2a) * 0.3
    x1s  = np.linspace(x1a - dx1, x1b + dx1, n)
    x2s  = np.linspace(x2a - dx2, x2b + dx2, n)
    for x1 in x1s:
        for x2 in x2s:
            if len(srcs) >= M:
                break
            srcs.append({"x1": x1, "x2": x2, "t": -T_ext})
    return srcs[:M]


# ──────────────────────────────────────────────
# 5. МАТРИЦЯ A  та псевдообернення
# ──────────────────────────────────────────────

def build_matrix(obs, srcs, c):
    """
    A[i, m] = G(obs[i], srcs[m])
    Рядки = точки спостережень (початкові + крайові)
    Стовпці = точки джерел (моделюючі)
    """
    N = len(obs)
    M = len(srcs)
    A = np.zeros((N, M))
    for i, o in enumerate(obs):
        for m, s in enumerate(srcs):
            A[i, m] = green(o["x1"], o["x2"], o["t"],
                            s["x1"], s["x2"], s["t"], c)
    return A


def solve(A, Y, lam=1e-2):
    """
    Псевдообернення: A·u = Y
    Метод: регуляризований МНК через розширену систему
      [A; √λ·I] u = [Y; 0]  →  мінімізує ‖Au-Y‖² + λ‖u‖²
    Повертає: u, rank, residual_norm
    """
   y_max = np.max(np.abs(Y))
    Y_norm = Y / y_max if y_max > 1e-10 else Y
    
    A_ext = np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])])
    Y_ext = np.concatenate([Y_norm, np.zeros(A.shape[1])])
    
    u, _, rank, _ = np.linalg.lstsq(A_ext, Y_ext, rcond=None)
    
    # Повертаємо масштаб назад
    u = u * y_max 
    
    residual = np.linalg.norm(A @ u - Y)
    return u, rank, residual


def reconstruct(x1_grid, x2_grid, t_val, srcs, u, c):
    """
    y'(x1, x2, t) = Σ_m  G(s, s'm) · u_m

    x1_grid, x2_grid — 1-D масиви координат сітки
    t_val            — момент часу
    Повертає матрицю y' розміром len(x1_grid) × len(x2_grid)
    """
    X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing='ij')
    Yp = np.zeros_like(X1)
    for m, s in enumerate(srcs):
        Yp += green(X1, X2, t_val, s["x1"], s["x2"], s["t"], c) * u[m]
    return Yp

def accuracy(Y_exact, Y_model):
    """
    Повертає: norm_abs, norm_rel (%), max_err
    """
    diff     = Y_exact - Y_model
    norm_abs = np.sqrt(np.mean(diff**2))        # RMS-норма
    norm_ref = np.sqrt(np.mean(Y_exact**2))
    norm_rel = (norm_abs / norm_ref * 100) if norm_ref > 1e-14 else 0.0
    max_err  = np.max(np.abs(diff))
    return norm_abs, norm_rel, max_err
