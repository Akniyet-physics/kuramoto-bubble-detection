import numpy as np, matplotlib.pyplot as plt
from scipy.signal import hilbert

def make(n, T, peak, crash=True, seed=42):
    rng = np.random.default_rng(seed)
    P, d = np.ones((T, n)) * rng.uniform(5,50,n), rng.normal(5e-4,8e-4,n)
    for t in range(1, T):
        s = np.clip((t-peak//2)/(peak//2),0,1)
        h = s**2 * np.sin(np.pi*t/T) * 0.5
        c = -0.01*np.exp(3*np.clip((t-peak)/(T-peak),0,1)) if crash and t>peak else 0
        P[t] = P[t-1] * np.exp(d + h*0.6 + rng.normal(0,0.02,n) + c)
    return P

def R(P):
    r = np.diff(np.log(P), axis=0)
    ph = np.array([np.unwrap(np.angle(hilbert(x - np.convolve(x,np.ones(60)/60,'same')))) for x in r.T]).T
    return np.abs(np.mean(np.exp(1j*ph), axis=1))

S  = lambda x: np.convolve(x, np.ones(30)/30, 'same')

Rd = S(R(make(8, 1500, 1280)))
Ra = S(R(make(9,  800,  600, crash=False, seed=99)))

c = Rd[:int(len(Rd)*0.9)].argmax()

fig, (a1,a2) = plt.subplots(1,2, figsize=(14,5), facecolor="#0A1628")
for a in (a1,a2):
    a.set_facecolor("#0D1B2A"); a.tick_params(colors="#7EB8D4"); a.grid(color="#1E3A5F",ls="--",alpha=0.5)

a1.plot(Rd, color="#00A8A8", lw=1.8, label="R(t) dotcoms")
a1.axvline(c, color="#EF5350", ls="--", label=f"Peak synchronization (day {c})")
a1.axhline(Rd[c], color="#F9A825", ls=":", label=f"R_crash={Rd[c]:.2f}")
a1.set(title="Dotcoms 1995–2001", ylim=(0,1.05)); a1.set_title(a1.get_title(), color="white")
a1.legend(fontsize=8, facecolor="#0D1B2A", labelcolor="white")

a2.plot(Ra, color="#F9A825", lw=1.8, label="R(t) AI")
a2.axhline(Rd[c], color="#EF5350", ls="--", label=f"Level of crash={Rd[c]:.2f}")
a2.axhline(Ra[-1], color="#66BB6A", ls=":", label=f"Now={Ra[-1]:.2f}")
a2.set(title="AI-market 2023–2026", ylim=(0,1.05)); a2.set_title(a2.get_title(), color="white")
a2.legend(fontsize=8, facecolor="#0D1B2A", labelcolor="white")

gap = Rd[c] - Ra[-1]
fig.suptitle(f"AI now: R={Ra[-1]:.2f}  |  Dotcom crash: R={Rd[c]:.2f}  |  Difference: {gap:.2f}", color="#F9A825")
plt.tight_layout()
plt.savefig("kuramoto.png", dpi=150, bbox_inches="tight", facecolor="#0A1628")
plt.show()