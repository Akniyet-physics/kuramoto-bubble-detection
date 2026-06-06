import numpy as np, matplotlib.pyplot as plt
from scipy.signal import hilbert
import yfinance as yf

def R(tickers, start, end):
    P = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    P = P.ffill().dropna(axis=1, how="all").dropna()
    print(f"  Используются: {list(P.columns)}  ({len(P)} дней)")
    if P.empty or P.shape[1] < 2:
        raise ValueError("Слишком мало данных — проверь тикеры или даты")
    ret = np.diff(np.log(P.values), axis=0)
    phases = np.array([np.unwrap(np.angle(hilbert(r - np.convolve(r, np.ones(60)/60, 'same')))) for r in ret.T]).T
    return P.index[1:], np.abs(np.mean(np.exp(1j * phases), axis=1))

S = lambda x: np.convolve(x, np.ones(30)/30, 'same')

print("Загружаем доткомы (1996–2002)...")
d_idx, Rd = R(["AMZN","EBAY","CSCO","INTC","ORCL","MSFT","DELL","AMAT"], "1996-01-01", "2002-01-01")

print("Загружаем ИИ-рынок (2023–сейчас)...")
a_idx, Ra = R(["PLTR","AI","SOUN","BBAI","SMCI"], "2023-01-01", "2026-01-01")

Rd, Ra = S(Rd), S(Ra)
crash = Rd.argmax()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0A1628")
for ax in (ax1, ax2):
    ax.set_facecolor("#0D1B2A"); ax.tick_params(colors="#7EB8D4", rotation=25)
    ax.grid(color="#1E3A5F", ls="--", alpha=0.5)

ax1.plot(d_idx, Rd, color="#00A8A8", lw=1.8, label="R(t) доткомы")
ax1.axvline(d_idx[crash], color="#EF5350", ls="--", label=f"Пик ({d_idx[crash].strftime('%Y-%m')})")
ax1.axhline(Rd[crash], color="#F9A825", ls=":", label=f"R_crash={Rd[crash]:.2f}")
ax1.set(title="Доткомы 1996–2001", ylim=(0,1.05)); ax1.set_title(ax1.get_title(), color="white")
ax1.legend(fontsize=8, facecolor="#0D1B2A", labelcolor="white")

ax2.plot(a_idx, Ra, color="#F9A825", lw=1.8, label="R(t) ИИ")
ax2.axhline(Rd[crash], color="#EF5350", ls="--", label=f"Уровень краха={Rd[crash]:.2f}")
ax2.axhline(Ra[-1], color="#66BB6A", ls=":", label=f"Сейчас={Ra[-1]:.2f}")
ax2.set(title="ИИ-рынок 2023–2026", ylim=(0,1.05)); ax2.set_title(ax2.get_title(), color="white")
ax2.legend(fontsize=8, facecolor="#0D1B2A", labelcolor="white")

fig.suptitle(f"ИИ сейчас: R={Ra[-1]:.2f}  |  Dotcom crash: R={Rd[crash]:.2f}  |  Разрыв: {Rd[crash]-Ra[-1]:.2f}", color="#F9A825")
plt.tight_layout(); plt.savefig("kuramoto_real.png", dpi=150, bbox_inches="tight", facecolor="#0A1628"); plt.show()