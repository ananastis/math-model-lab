"""
Лабораторна робота 3
Робоча група 5: Булига Анастасія та Носкова Каріна
model.py — математичне ядро лабораторної роботи 
Хвильове рівняння в R², простір [0,T]
L(∂s) = ∂²t − c²(∂²x1 + ∂²x2)
G(s)  = H(t − r/c) / (2πc√(c²t² − r²))
"""

import numpy as np


# ТЕСТОВІ ФУНКЦІЇ y(x1, x2, t)

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


# ФУНКЦІЯ ГРІНА  G(s, s')

def green(x1, x2, t, x1s, x2s, ts, c, eps=1e-3):
    # Примусово конвертуємо у numpy (вирішує 'float has no ndim')
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    t  = np.asarray(t,  dtype=float)

    dt   = t - float(ts)
    r2   = (x1 - float(x1s))**2 + (x2 - float(x2s))**2
    disc = c**2 * dt**2 - r2

    if x1.ndim == 0:               # scalar
        if float(dt) <= 0 or float(disc) <= eps:
            return 0.0
        return float(1.0 / (2.0 * np.pi * c * np.sqrt(float(disc))))
    else:                          # array
        mask = (dt > 0) & (disc > eps)
        out  = np.zeros(x1.shape, dtype=float)
        out[mask] = 1.0 / (2.0 * np.pi * c * np.sqrt(disc[mask]))
        return out


# ТОЧКИ СПОСТЕРЕЖЕНЬ  (початкові + крайові)

def make_observations(fn, c, x1a, x1b, x2a, x2b, T, R0, Rg):
    obs = []
    n = max(2, int(np.ceil(np.sqrt(R0))))
    x1_grid = np.linspace(x1a, x1b, n)
    x2_grid = np.linspace(x2a, x2b, n)
    for x1 in x1_grid:
        for x2 in x2_grid:
            if len([o for o in obs if o["type"]=="init"]) >= R0:
                break
            val = fn(x1, x2, 0.0, c)
            obs.append({"type":"init", "x1":x1, "x2":x2, "t":0.0, "Y":val})

    n_b = max(2, Rg // 4)
    t_grid = np.linspace(0, T, n_b)
    for t_ in t_grid:
        for x2 in np.linspace(x2a, x2b, n_b):
            obs.append({"type":"bound", "x1":x1a, "x2":x2, "t":t_, "Y": fn(x1a, x2, t_, c)})
        for x2 in np.linspace(x2a, x2b, n_b):
            obs.append({"type":"bound", "x1":x1b, "x2":x2, "t":t_, "Y": fn(x1b, x2, t_, c)})
        for x1 in np.linspace(x1a, x1b, n_b):
            obs.append({"type":"bound", "x1":x1, "x2":x2a, "t":t_, "Y": fn(x1, x2a, t_, c)})
        for x1 in np.linspace(x1a, x1b, n_b):
            obs.append({"type":"bound", "x1":x1, "x2":x2b, "t":t_, "Y": fn(x1, x2b, t_, c)})

    Y = np.array([o["Y"] for o in obs])
    return obs, Y


# ТОЧКИ ДЖЕРЕЛ  s'm  (зовнішня область t < 0)

def make_sources(x1a, x1b, x2a, x2b, M, c=1.0, T=1.0):
    r_max = np.sqrt((x1b - x1a)**2 + (x2b - x2a)**2)
    T_ext = r_max / c + T + 1.0
    
    srcs = []
    n = max(2, int(np.ceil(np.sqrt(M))))
    
    margin = 0.5
    x1s = np.linspace(x1a - margin, x1b + margin, n)
    x2s = np.linspace(x2a - margin, x2b + margin, n)
    
    t_layers = np.linspace(-T_ext, -T_ext * 0.3, 4)
    
    for ts in t_layers:
        for x1 in x1s:
            for x2 in x2s:
                if len(srcs) >= M:
                    return srcs[:M]
                srcs.append({"x1": x1, "x2": x2, "t": float(ts)})
    return srcs[:M]

# МАТРИЦЯ A  та псевдообернення

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


def solve(A, Y, lam=0.01):
    # Нормалізація стовпців (усуває різницю масштабів між джерелами)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms < 1e-12] = 1.0  # захист від нулів
    A_norm = A / col_norms
    
    cond = np.linalg.cond(A_norm)
    print(f"Число обумовленості (після норм.): {cond:.2e}")
    
    U, s, Vt = np.linalg.svd(A_norm, full_matrices=False)
    s_inv = s / (s**2 + lam)
    u_norm = Vt.T @ (s_inv * (U.T @ Y))
    
    # Повертаємо до оригінального масштабу
    u = u_norm / col_norms
    
    residual = np.linalg.norm(A @ u - Y)
    return u, len(s), residual


def reconstruct(x1_grid, x2_grid, t_val, srcs, u, c):
  
    X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing='ij')
    Yp = np.zeros_like(X1)
    for m, s in enumerate(srcs):
        Yp += green(X1, X2, t_val, s["x1"], s["x2"], s["t"], c) * u[m]
    return Yp

def accuracy(Y_exact, Y_model):
    
    diff     = Y_exact - Y_model
    norm_abs = np.sqrt(np.mean(diff**2))        # RMS-норма
    norm_ref = np.sqrt(np.mean(Y_exact**2))
    norm_rel = (norm_abs / norm_ref * 100) if norm_ref > 1e-14 else 0.0
    max_err  = np.max(np.abs(diff))
    return norm_abs, norm_rel, max_err
