import numpy as np
import matplotlib.pyplot as plt

T     = 252
dt    = 1/252
s0    = 100 # Initial price
mu    = 0.1 # Expected return
sigma = 0.2 # Volatility
rho   = -0.2 # Correlation
kappa = 0.3 # Revert rate
theta = 0.2 # Long-term volatility
xi    = 0.2 # Volatility of instantaneous volatility
v0    = 0.2 # Initial instantaneous volatility


def sim_mc_Heston(s0, mu, v0, rho, kappa, theta, xi, T, dt):
    avg  = np.array([0, 0])
    cov = np.matrix([[1, rho], [rho, 1]])
    w   = np.random.multivariate_normal(avg, cov, T)
    w_s = w[:,0]*np.sqrt(dt)
    w_v = w[:,1]*np.sqrt(dt)

    vt = np.zeros(T)
    vt[0] = v0
    st    = np.zeros(T)
    st[0] = s0
    for t in range(1,T):
        vt[t] = np.abs(vt[t-1] + kappa*(theta-vt[t-1])*dt + \
                                    xi*np.sqrt(vt[t-1])*w_v[t])
        st[t] = st[t-1]*np.exp((mu - 0.5*vt[t-1])*dt + \
                                    np.sqrt(vt[t-1])*w_s[t])
    return st, vt

st, vt = sim_mc_Heston(s0, mu, v0, rho, kappa, theta, xi, T, dt)

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_ylabel('Stock Price', color=color)
ax1.plot(range(T), st, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Volatility', color=color)
ax2.plot(range(T), vt, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Stock Price and Volatility sample path with Heston model')
plt.show()
