## 📈 Dashboard de Séries Temporais — SARIMA (from scratch)

Dashboard interativo para análise e previsão de séries temporais, com implementação **manual de um modelo SARIMA** (sem bibliotecas prontas).

---

## 🚀 O que o projeto mostra

* 📊 Série histórica + ajuste (fitted) + previsão 2024
* 🔄 Detecção de sazonalidade via ACF
* 📉 Decomposição: tendência, sazonalidade e resíduos
* 🎯 Métricas: MAPE, RMSE, MAE, R²
* 🔮 Intervalo de confiança (95%)

---

## 🧠 Modelo

* SARIMA(1,1,1)(1,1,1)\₁₂ (quando sazonalidade detectada)
* Diferenciação dupla: tendência + sazonalidade
* Estimação via regressão OLS (ARMA simplificado)

---

## ⚙️ Tecnologias

* Python (NumPy, Pandas)
* HTML + CSS
* JavaScript
* Chart.js

---

## ▶️ Como executar

```bash
python script.py
```

Abra o arquivo:

```
ts_arima_dashboard.html
```

---

## 🎯 Objetivo

Demonstrar:

* Fundamentos de séries temporais
* Implementação de modelo estatístico do zero
* Construção de dashboard analítico

---

## 💡 Diferencial

Sem uso de bibliotecas prontas de forecasting (tipo statsmodels) — tudo implementado manualmente.

---
![SARIMA_from_scratch](https://raw.githubusercontent.com/gabrielcardoso-estatistico/SARIMA_from_scratch/refs/heads/main/Captura%20de%20tela%202026-03-26%20081055.png)

