import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

class PDEPricing():
    def __init__(self, I, J, scheme):
        self.I = I
        self.J = J
        self.scheme = scheme

    def get_explicit_coeff(self, s, r, σ, Δt, Δs):
        c = Δt * σ**2 * s**2 / (2 * Δs**2)
        b = 1 - 2 * c + Δt * r * (s/Δs - 1)
        a = c - Δt * r * s / Δs
        return a, b, c

    def get_implicit_coeff(self, s, r, σ, Δt, Δs):
        c = -Δt * σ**2 * s**2 / (2 * Δs**2)
        b = 1 - 2 * c - Δt * r * (s/Δs - 1)
        a = c + Δt * r * s / Δs
        return a, b, c

    def get_trans_mtx(self, **kwargs):
        d_schm = {'implicit': self.get_implicit_coeff,
                  'explicit': self.get_explicit_coeff}

        a, b, c = d_schm[self.scheme](**kwargs)

        diag = diags([a[1:],b,c[:-1]], offsets=[-1,0,1]).toarray()
        if self.scheme == 'implicit': diag = np.linalg.inv(diag)
        return diag[1:diag.shape[0]-1]

    def set_call_boundaries(self, s, t, K):
        top = s[-1] - np.exp(-r * (t[-1] - t)) * K
        bot = np.zeros(t.shape)
        fin = np.maximum(s - K, 0)
        return top, bot, fin

    def set_fwd_call_boundaries(self, s, t, pK, i1):
        def get_fin():
            fin = np.zeros(s.shape)
            for j in range(1, self.J):
                K = s[j] * (1 + pK)
                fin[j] = self.price_call(s[j], K, T, r, σ, t[i1])
            return fin

        fin = get_fin()
        top = np.exp(-r * (t[i1] - t[:i1])) * fin[-1]
        bot = np.zeros(t[:i1].shape)
        return top, bot, fin

    def price_call(self, s0, K, T, r, σ, t=0):
        t_min = t
        t_max = T
        s_min = 0
        s_max = 2 * K
        t = np.linspace(t_min, t_max, self.I)
        s = np.linspace(s_min, s_max, self.J)
        Δs = s[1]
        Δt = t[1] - t[0]

        v = np.zeros((self.I, self.J))

        v[:,-1], v[:,0], v[-1,:] = self.set_call_boundaries(s, t, K)

        M = self.get_trans_mtx(s=s, r=r, σ=σ, Δt=Δt, Δs=Δs)

        for i in range(self.I-1, 0, -1):
            v[i-1, 1:self.J-1] = M @ v[i,:]

        j = min([x for x in range(len(s)) if s[x]>=s0])
        return v[0, j] * s0/s[j]

    def price_fwd_call(self, s0, pK, t1, T, r, σ):
        t_min = 0
        t_max = T
        s_min = 0
        s_max = 2 * s0
        t = np.linspace(t_min, t_max, self.I)
        s = np.linspace(s_min, s_max, self.J)
        Δs = s[1] - s[0]
        Δt = t[1] - t[0]
        i1 = int(t1 * self.I)

        v = np.zeros((self.I, self.J))

        v[:i1,-1], v[:i1,0], v[i1,:] = self.set_fwd_call_boundaries(s,t,pK, i1)
        v = v[:i1+1, :]

        M = self.get_trans_mtx(s=s, r=r, σ=σ, Δt=Δt, Δs=Δs)

        for i in range(v.shape[0]-1, 0, -1):
            v[i-1, 1:self.J-1] = (M @ v[i,:])

        j = min([x for x in range(len(s)) if s[x]>=s0])
        return v[0, j]*s0/s[j]

if __name__=='__main__':
    scheme = 'implicit'
    I = 60
    J = 60
    r = .03
    σ = .2
    K = 100
    T = 1
    s0 = 100
    pK = .0
    t1 = .5
    pde = PDEPricing(I, J, scheme)

    p = pde.price_call(s0, K, T, r, σ, t=0)
    print('Vanilla Call: ', p)

    #p = pde.price_fwd_call(s0, pK, t1, T, r, σ)
    #print('Forward Call: ', p)


    p = []
    for t1 in range(0, 100, 10):
        t1 /= 100
        p.append(pde.price_fwd_call(s0, pK, t1, T, r, σ))
    print(p)
    plt.plot(p)
    plt.show()
