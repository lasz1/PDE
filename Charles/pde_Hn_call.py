import numpy as np


class PDEHnPricing():
    def __init__(self, N, I, J, scheme):
        self.N = N
        self.I = I
        self.J = J
        self.scheme = scheme

    def get_trans_mtx(self, s, v, Δt, Δs, Δv, r, σ, ρ, κ, θ, λ):
        M = np.empty((self.I, self.J), dtype=np.ndarray)
        for i in range(self.I):
            for j in range(self.J):
                α = κ * (θ - v[j]) - λ * σ * np.sqrt(v[j])
                a = Δt / Δs / Δv * ρ * s[i]* v[j] / 4
                b = Δt / Δs**2 * v[j] * s[i]**2 / 2
                c = Δt / Δv**2 * v[j] * σ**2 / 2
                d = Δt / Δs * s[i] * (r + s[i] * v[j] /2)
                e = Δt / Δv * (σ**2 * v[j] / 2 / Δv + α)
                f = 1 - 2 * (a + c) - Δt * (r * (1 + s[i] / Δs) + α / Δv)
                M[i, j] = np.array([[a, b, a], [c, f, e], [a, d, a]])
        return M

    def set_call_boundaries(self, u_n, s, z, K, r):
        u_n[0,:]  = np.zeros(self.J)
        u_n[-1,:] = np.ones(self.J) * np.maximum(s[-1] - K, 0) * np.exp(-r * z)
        u_n[:,0]  = np.maximum(s - K, 0) * np.exp(-r * z)
        u_n[:,-1] = s
        return u_n

    def price_call(self, s0, K, T, v0, r, σ, ρ, κ, θ, λ, t=0):
        t_min = t
        t_max = T
        s_min = 0
        s_max = 2 * K
        #s_max = K * np.exp(8 * np.sqrt(v0) * np.sqrt(T))
        v_min = 0
        v_max = v0 * np.exp(8 * σ * np.sqrt(T))
        τ = np.linspace(t_min, t_max, self.N)
        s = np.linspace(s_min, s_max, self.I)
        v = np.linspace(v_min, v_max, self.J)
        Δs = s[1]
        Δv = v[1]
        Δt = τ[1] - τ[0]

        M = self.get_trans_mtx(s, v, Δt, Δs, Δv, r, σ, ρ, κ, θ, λ)

        for n in τ[::-1]:
            if n == t_max:
                u_N = np.array([np.maximum(s - K, 0),]*self.J).transpose()
            else:
                u_n = np.zeros((self.I, self.J))
                z = n * Δt
                u_n = self.set_call_boundaries(u_n, s, z, K, r)
                for i in range(1, self.I - 1):
                    for j in range(1, self.J - 1):
                        u_n[i, j] = np.sum(M[i, j] * u_N[i-1:i+2, j-1:j+2])
                u_N = u_n
        i = min([x for x in range(len(s)) if s[x]>=s0])
        j = min([x for x in range(len(s)) if v[x]>=v0])
        return u_N[i, j] * s0/s[j]


if __name__ == '__main__':
    scheme = 'explicit'
    N  = 10
    I  = 10
    J  = 10
    s0 = 100
    K  = 110
    T  = 1
    v0 = .04
    r  = .03
    σ  = .2
    ρ  =-.7
    κ  = .8
    θ  = .04
    λ  = 0

    pde = PDEHnPricing(N, I, J, scheme)
    p = pde.price_call(s0, K, T, v0, r, σ, ρ, κ, θ, λ)
    print(p)
