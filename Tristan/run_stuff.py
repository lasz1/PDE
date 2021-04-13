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
                a = σ * ρ * s[i] * v[j] /( 4 * Δs * Δv)
                b = Δt * (s[i]**2 * v[j] / (2*(Δs**2)) + r * s[i] / (Δv**2))
                d = Δt * (s[i]**2 * v[j] / (2*(Δs**2)) - r * s[i] / (Δv**2))
                c = Δt * (σ**2 * v[j] / (2*(Δv**2) - κ*(θ - v[j])/(2*Δv))
                e = Δt * (σ**2 * v[j] / (2*(Δv**2) + κ*(θ - v[j])/(2*Δv))
                f = 1 - Δt * (s[i]**2 * v[j] / (Δs**2) + σ**2 * v[j] / ((Δv**2))
                M[i, j] = np.array([[a, b, -a], [c, f, e], [-a, d, a]])
        return M

    def set_call_boundaries(self, u_n, s, z, K, r, Δs):
        u_n[0,:]  = np.zeros(self.J)
        u_n[-1,:] = np.ones(self.J) * np.maximum(s[-1] - K * np.exp(-r * z), 0)
        u_n[-2,:] = u_n[-1,:] - Δs
        u_n[:,0]  = np.maximum(s - K * np.exp(-r * z), 0)
        u_n[:,-1] = s
        return u_n

    def set_fwd_call_boundaries(self, s, v, pK, t1, T, r ,σ, ρ, κ, θ, λ):
        L = np.zeros((len(s), len(v)))
        for i in range(1, len(s)):
            for j in range(1, len(v)):
                K = s[j] * (1 + pK)
                L[i, j] = self.price_call(s[i], K, T, v[j], r, σ, ρ, κ, θ, λ, t1)
        return L

    def price_call(self, s0, K, T, v0, r, σ, ρ, κ, θ, λ, t=0):
        t_min = t
        t_max = T
        s_min = 0
        s_max = s0*  np.exp(8 * v0 * np.sqrt(T))
        #s_max = K * np.exp(8 * np.sqrt(v0) * np.sqrt(T))
        v_min = 0
        v_max = v0 * np.exp(8 * σ * np.sqrt(T))
        v_max = 2 * v0 
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
                u_n = self.set_call_boundaries(u_n, s, z, K, r, Δs)
                for i in range(1, self.I - 2):
                    for j in range(1, self.J - 1):
                        u_n[i, j] = np.sum(M[i, j] * u_N[i-1:i+2, j-1:j+2])
                u_N = u_n
        #print(u_N)
        i_u = min([x for x in range(len(s)) if s[x]>=s0])
        i_d = max([x for x in range(len(s)) if s[x]<=s0])
        j_u = min([x for x in range(len(v)) if v[x]>=v0])
        j_d = max([x for x in range(len(v)) if v[x]<=v0])
        u0 = np.mean([u_N[i_u, j_u],u_N[i_u, j_d],u_N[i_d, j_u],u_N[i_d, j_d]])
        return u0

    def price_fwd_call(self, s0, v0, pK, t1, T, r ,σ, ρ, κ, θ, λ):
        t_min = 0
        t_max = T
        s_min = 0
        s_max = 2 * K
        #s_max = K * np.exp(8 * np.sqrt(v0) * np.sqrt(T))
        v_min = 0
        v_max = v0 * np.exp(8 * σ * np.sqrt(T))
        v_max = 2 * v0
        τ = np.linspace(t_min, t_max, self.N)
        s = np.linspace(s_min, s_max, self.I)
        v = np.linspace(v_min, v_max, self.J)
        Δs = s[1]
        Δv = v[1]
        Δt = τ[1] - τ[0]
        n1 = int(t1 * self.N)

        M = self.get_trans_mtx(s, v, Δt, Δs, Δv, r, σ, ρ, κ, θ, λ)
        for n in τ[:n1+1][::-1]:
            if n == τ[n1]:
                u_N = self.set_fwd_call_boundaries(s, v, pK, t1, T, r ,σ, ρ, κ, θ, λ)
            else:
                u_n = np.zeros((self.I, self.J))
                z = n * Δt
                u_n = self.set_call_boundaries(u_n, s, z, K, r, Δs)
                for i in range(1, self.I - 2):
                    for j in range(1, self.J - 1):
                        u_n[i, j] = np.sum(M[i, j] * u_N[i-1:i+2, j-1:j+2])
                u_N = u_n

        i_u = min([x for x in range(len(s)) if s[x]>=s0])
        i_d = max([x for x in range(len(s)) if s[x]<=s0])
        j_u = min([x for x in range(len(v)) if v[x]>=v0])
        j_d = max([x for x in range(len(v)) if v[x]<=v0])
        u0 = np.mean([u_N[i_u, j_u],u_N[i_u, j_d],u_N[i_d, j_u],u_N[i_d, j_d]])
        return u0



if __name__ == '__main__':
    scheme = 'explicit'
    N  = 10
    I  = 10
    J  = 20
    s0 = 100
    K  = 100
    T  = 1
    v0 = .04
    r  = 0
    σ  = 0
    ρ  =-.7
    κ  = 0
    θ  = .04
    λ  = 0
    pK = 0
    t1 = 0

    pde = PDEHnPricing(N, I, J, scheme)
    p = pde.price_call(s0, K, T, v0, r, σ, ρ, κ, θ, λ)
    print(p)
    p = pde.price_fwd_call(s0, v0, pK, t1, T, r ,σ, ρ, κ, θ, λ)
    print(p)
