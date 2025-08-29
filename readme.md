# Pipeline HFT — Build & Run (mínimo)

## Compilar
```bash
g++ -std=gnu++17 -O2 process_market.cpp   -o process_market
g++ -std=gnu++17 -O2 get_nowcast.cpp      -o get_nowcast
g++ -std=gnu++17 -O2 mlp_infer_plain.cpp  -o mlp_infer_plain


# 1) Generar df_all.csv desde ./market_data
```
./process_market --dir ./market_data
```

# 2) Generar xy_train.csv (features/label) y estimar p_next_hat
```bash
./get_nowcast --df df_all.csv --target "AL30_1205_CI_CCL" --k_last 3 --top_others 4 --dt_median_window 20 --xy_out xy_train.csv

```

# 3) Inferencia/Evaluación MLP
```bash
./mlp_infer_plain mlp_bundle.txt --eval xy_train.csv    # eval: MSE, R2, 5 primeras
./mlp_infer_plain mlp_bundle.txt X_only.csv             # solo predicciones (una por línea)
```


# Resumen — `process_market.cpp`

- Lee CSVs de `./market_data` (`fecha_nano, price, quantity, side`) por instrumento.
- Agrupa por `fecha_nano` y `side` (BI/OF/TRADE).
- Calcula **VWAP** (∑p·q/∑q) y **spread** (desvío estándar ponderado).
- Escribe `df_all.csv` con: `instrument, side, fecha_nano, ts_sec, vwap, spread`.

---

# Resumen — `predict_next.cpp`

- Desde `df_all.csv` (solo `TRADE`): usa `--target` y agrega `--top_others` más líquidos (si `0`, es univariado).
- Para cada `t0`: recta con `k_last` → features `p_i` (precio) y `m_i` (pendiente); **target** `m_next` (pendiente siguiente).
- Escribe `xy_train.csv` (para entrenar la red).
- Estima `p_next_hat = p0 + m_hat·dt_hat` y devuelve JSON (`selected_instruments, n_samples, p0, m_hat, dt_hat_sec, p_next_hat`).

---

# Resumen — `mlp_infer_plain.cpp`

- Carga `mlp_bundle.txt` (activación, `n_features`, scaler, capas `W/b`).
- Lee `--eval xy_train.csv` (con `y`) o `X_only.csv` (solo features).
- Estandariza con el scaler del bundle y hace forward MLP (activación en ocultas, salida lineal).
- `--eval`: reporta **MSE** y **R²** + 5 predicciones; con `X_only.csv`: una predicción por línea.
