import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import yfinance as yf

# =============================================================
# KURAMOTO SYNCHRONIZATION MODEL — REAL DATA VERSION
# =============================================================

# Tickers used
DOTCOM_TICKERS = ["AMZN", "EBAY", "CSCO", "INTC", "ORCL", "MSFT", "DELL", "AMAT"]
AI_TICKERS     = ["PLTR", "AI", "SOUN", "BBAI", "SMCI"]

DOTCOM_START = "1996-01-01"
DOTCOM_END   = "2002-01-01"
AI_START     = "2023-01-01"
AI_END       = "2026-01-01"

SMOOTHING_WINDOW = 30  # days for moving average
PHASE_WINDOW     = 60  # days for trend removal

# --- Function: Kuramoto order parameter R (real data) ---
# Downloads real price data via yfinance.
# Computes daily log-returns, extracts phases using Hilbert transform,
# and returns the synchronization parameter R (0 to 1).

def R(tickers, start, end):
    P = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    P = P.ffill().dropna(axis=1, how="all").dropna()
    print(f"  Using: {list(P.columns)}  ({len(P)} days)")
    if P.empty or P.shape[1] < 2:
        raise ValueError("Too little data — check tickers or dates")
    ret = np.diff(np.log(P.values), axis=0)
    phases = np.array([np.unwrap(np.angle(hilbert(r - np.convolve(r, np.ones(60)/60, 'same')))) for r in ret.T]).T
    return P.index[1:], np.abs(np.mean(np.exp(1j * phases), axis=1))

# --- Smoothing ---
# Removes noise by averaging over a 30-day window.

S = lambda x: np.convolve(x, np.ones(30)/30, 'same')

print("Downloading dotcoms (1995–2001)...")
d_idx, Rd = R(DOTCOM_TICKERS, DOTCOM_START, DOTCOM_END)

print("Downloading AI-market (2023–now)...")
a_idx, Ra = R(AI_TICKERS, AI_START, AI_END)

Rd, Ra = S(Rd), S(Ra)

# --- Detect synchronization peak ---
# Find the day when dot-com companies reached maximum synchronization —
# the transition point from independent behaviour to herd effect.

crash_day = Rd.argmax()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0A1628")
for ax in (ax1, ax2):
    ax.set_facecolor("#0D1B2A") 
    ax.tick_params(colors="#7EB8D4", rotation=25)
    ax.grid(color="#1E3A5F", ls="--", alpha=0.5)

ax1.plot(d_idx, Rd, color="#00A8A8", lw=1.8, label="R(t) dotcoms")
ax1.axvline(d_idx[crash_day], color="#EF5350", ls="--", label=f"Peak synchronization ({d_idx[crash_day].strftime('%Y-%m')})")
ax1.axhline(Rd[crash_day], color="#F9A825", ls=":", label=f"R_crash={Rd[crash_day]:.2f}")
ax1.set(title="Dotcoms 1995–2001", ylim=(0,1.05)) 
ax1.set_title(ax1.get_title(), color="white")
ax1.legend(fontsize=8, facecolor="#0D1B2A", labelcolor="white")

ax2.plot(a_idx, Ra, color="#F9A825", lw=1.8, label="R(t) AI")
ax2.axhline(Rd[crash_day], color="#EF5350", ls="--", label=f"Level of crash={Rd[crash_day]:.2f}")
ax2.axhline(Ra[-1], color="#66BB6A", ls=":", label=f"Now={Ra[-1]:.2f}")
ax2.set(title="AI-market 2023–2026", ylim=(0,1.05)) 
ax2.set_title(ax2.get_title(), color="white")
ax2.legend(fontsize=8, facecolor="#0D1B2A", labelcolor="white")

fig.suptitle(f"AI now: R={Ra[-1]:.2f}  |  Dotcom crash: R={Rd[crash_day]:.2f}  |  Difference: {Rd[crash_day]-Ra[-1]:.2f}", color="#F9A825")
plt.tight_layout() 
plt.savefig("kuramoto_real.png", dpi=150, bbox_inches="tight", facecolor="#0A1628")
plt.show()