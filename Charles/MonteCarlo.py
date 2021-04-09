import numpy as np
from scipy.stats import norm

class MonteCarloPricing():
    def __init__(self, N, p_conf=.95, M=252):
        self.N = int(N)
        self.p_conf = p_conf
        self.M = M

    def gen_BS_prices(self, s0, r, σ, T, t=0, **kwargs):
        dt, nd =(T - t) / self.M, (T - t) * self.M
        w = np.zeros(nd)
        s = np.ones(nd) * s0
        for i in range(1, nd):
            w[i] = w[i-1] + np.random.normal() * np.sqrt(dt)
            s[i] = s0 * np.exp((r - σ**2 / 2) * dt + σ * w[i])
        return s

    def gen_Hn_prices(self, s0, r, v0, κ, θ, γ, ρ, T, t=0, **kwargs):
        dt, nd =(T - t) / self.M, (T - t) * self.M
        avg  = np.array([0, 0])
        cov = np.matrix([[1, ρ], [ρ, 1]])
        w   = np.random.multivariate_normal(avg, cov, nd)
        w_s = w[:,0]*np.sqrt(dt)
        w_v = w[:,1]*np.sqrt(dt)

        v = np.zeros(nd)
        v[0] = v0
        s = np.zeros(nd)
        s[0] = s0
        for t in range(1,nd):
            v[t] = np.abs(v[t-1] + κ*(θ-v[t-1])*dt + γ*np.sqrt(v[t-1])*w_v[t])
            s[t] = s[t-1]*np.exp((r - v[t]/2)*dt + np.sqrt(v[t])*w_s[t])
        return s

    def get_payoff_euro_call(self, s, K, **kwargs):
        return max(s[-1] - K, 0)

    def get_payoff_fwrd_call(self, s, τ, pK, **kwargs):
        K = s[τ] * (1 + pK)
        return max(s[-1] - K, 0)

    def sim_MC(self, gen_fct, payoff_fct, **kwargs):
        def get_res():
            s = gen_fct(**kwargs)
            return payoff_fct(s, **kwargs)

        res = np.zeros(self.N)
        for n in range(self.N):
            res[n] = get_res()
        return res

    def price(self, gen_fct, payoff_fct, **kwargs):
        z_score = abs(norm.ppf((1 - self.p_conf) / 2))
        MC_res = self.sim_MC(gen_fct, payoff_fct, **kwargs)
        exp = np.mean(MC_res)
        var = np.var(MC_res)
        conf_margin = z_score * np.sqrt(var / self.N)
        str = '%.2f' % exp + '$ ± ' +  '%.2f' % conf_margin
        return str




if __name__ == '__main__':
    ########### Monte Carlo ###########
    N = 1e4         # Number of sample paths

    ########### Price Dynamics ###########
    # Black and Scholes Parameters:
    T = 1         # Maturity in years
    x = 100       # Initial price
    r = .03       # Drift / Risk free return
    σ = .2        # Volatility

    # Heston Parameters:
    T = 1         # Maturity in years
    s0 = 100      # Initial price
    r  = .01      # Drift / Risk free return
    v0 = .05      # Initial Variance
    κ  = .3       # Mean reversion factor of the variance
    θ  = .1       # Mean of the variance
    γ  = .15      # Volatility of the variance
    ρ  = -.2      # Leverage factor

    ########### Option Types ###########
    # European Call Parameters:
    K = 110       # Stike price

    # For Call Parameters:
    τ = 6 * 21   # Start date
    pK = +.0     # Percentage above or lower than the price at τ

    mc = MonteCarloPricing(N)

    # European Call
    #print(' ------- European Call -------')
    #payoff_fct = mc.get_payoff_euro_call

    ## Black & Scholes
    #gen_fct = mc.gen_BS_prices
    #x = mc.price(gen_fct, payoff_fct, s0=s0, r=r, σ=σ, T=T, K=K)
    #print('\t Black & Scholes: ', x)

    ## Heston
    #gen_fct = mc.gen_Hn_prices
    #x = mc.price(gen_fct, payoff_fct, s0=s0, r=r, v0=v0, κ=κ, θ=θ, γ=γ, ρ=ρ, T=T, K=K)
    #print('\t Heston: ', x)


    # Forward Call
    print('\n ------- Forward Call -------')
    payoff_fct = mc.get_payoff_fwrd_call

    ## Black & Scholes
    gen_fct = mc.gen_BS_prices
    x = mc.price(gen_fct, payoff_fct, s0=s0, r=r, σ=σ, T=T, τ=τ, pK=pK)
    print('\t Black & Scholes: ', x)

    ## Heston
    #gen_fct = mc.gen_Hn_prices
    #x = mc.price(gen_fct, payoff_fct, s0=s0, r=r, v0=v0, κ=κ, θ=θ, γ=γ, ρ=ρ, T=T, τ=τ, pK=pK)
    #print('\t Heston: ', x)
