import h5py
import numpy as np
from scipy.special import factorial
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from torchdeepretina.pyret_func import estfr

class distribution:
    
    def __init__(self, truncation):
        self.t = truncation

    def truncated_gaussian(self, n, n0, s):
        summation = 0
        for i in range(self.t):
            summation += np.exp(-(i-n0)**2/2/s**2)
        if n >= self.t:
            return 0
        else:
            return np.exp(-(n-n0)**2/2/s**2)/summation

    def sigmoid_para(self, x, a, b, c):
        return a/(1+1/np.exp((x-b)/c))

    def inverse_sigmoid_para(self, x, a, b, c):
        if x == 0:
            return self.inverse_sigmoid_para(0.001, a, b, c)
        else:
            return b - c * np.log(a/x-1)

    def poisson_scale1(self, n, r, k):
        p = 0
        for i in range(np.ceil(k*(n-1)).astype(np.int), np.floor(k*(n+1)).astype(np.int)+1):
            if i >= 0:
                p += np.exp(-r*k)*(r*k)**i/factorial(i)*(1-np.abs(i/k-n))
        return p
    
    def poisson_scale2(self, n, r, k):
        p = np.exp(-r*k)*(r*k)**(n*k)/factorial(n*k)
        return p

    def poisson_scale3(self, n, r, k):
        p = 0
        for i in range(np.ceil(k*(n-0.5)).astype(np.int), np.ceil(k*(n+0.5)-1).astype(np.int)+1):
            if i >= 0:
                p += np.exp(-r*k)*(r*k)**i/factorial(i)
        return p

    def truncated_poisson(self, n, r, k, p_version):
        poisson_func = getattr(self, 'poisson_scale'+str(p_version))
        summation = 0
        for i in range(self.t):
            summation += poisson_func(i, r, k)
        if n >= self.t:
            return 0
        else:
            return poisson_func(n, r, k)/summation

    def binomial_scale(self, n, p, k):
        M = self.t - 1
        #n = min(n, M)
        summation = np.sum([factorial(k*M)/factorial(k*i)/factorial(k*M-k*i)*p**(k*i)*(1-p)**(k*M-k*i) for i in range(M+1)])
        pr = factorial(k*M)/factorial(k*n)/factorial(k*M-k*n)*p**(k*n)*(1-p)**(k*M-k*n) / summation
        return pr
    
    def mean(self, dist_name, r, k, **kwargs):
        dist_func = getattr(self, dist_name)
        return np.sum([dist_func(i, r, k, **kwargs)*i for i in range(self.t)])
    
    def var(self, dist_name, r, k, **kwargs):
        dist_func = getattr(self, dist_name)
        mean = self.mean(dist_name, r, k, **kwargs)
        return np.sum([dist_func(i, r, k, **kwargs)*(i-mean)**2 for i in range(self.t)])
    
    def rate2para(self, dist_name, k, rate, n_samples=100, **kwargs):
        
        dist_func = getattr(self, dist_name)

        if 'gaussian' in dist_name:
            r_list = np.linspace(-self.t, 2*self.t, n_samples)
        elif 'poisson' in dist_name:
            r_list = np.linspace(0, 2*self.t, n_samples)
        elif 'binomial' in dist_name:
            r_list = np.linspace(0, 1, n_samples)
            
        mean_list = [self.mean(dist_name, r, k, **kwargs) for r in r_list]
        if 'gaussian' in dist_name:
            a,b,c = curve_fit(self.sigmoid_para, r_list, mean_list)[0]
            r = self.inverse_sigmoid_para(rate/100, a, b, c)
        else:
            f = interp1d(mean_list, r_list)
            r = max(f(rate/100), 0)

        return r
    
    def log_likelihood(self, dist_name, k, single_trial_bin, cell, n_samples=100, **kwargs):
        
        dist_func = getattr(self, dist_name)

        if 'gaussian' in dist_name:
            r_list = np.linspace(-self.t, 2*self.t, n_samples)
        elif 'poisson' in dist_name:
            r_list = np.linspace(0, 2*self.t, n_samples)
        elif 'binomial' in dist_name:
            r_list = np.linspace(0, 1, n_samples)
            
        mean_list = [self.mean(dist_name, r, k, **kwargs) for r in r_list]
        if 'gaussian' in dist_name:
            a,b,c = curve_fit(self.sigmoid_para, r_list, mean_list)[0]
        else:
            f = interp1d(mean_list, r_list)

        ll = 0
        for time in range(single_trial_bin.shape[1]):
            rate = single_trial_bin[:,time,cell].mean()
            if 'gaussian' in dist_name:
                r = self.inverse_sigmoid_para(rate, a, b, c)
            else:
                r = max(f(rate), 0)
            for trial in range(single_trial_bin.shape[0]):
                n = single_trial_bin[trial,time,cell]
                ll += np.log(dist_func(n, r, k, **kwargs))
        return ll
    
    def optimize_k(self, dist_name, single_trial_bin, cell, a=1, b=10, **kwargs):

        assert np.sign(self.log_likelihood(dist_name, a, single_trial_bin, cell, **kwargs) - self.log_likelihood(dist_name, a+0.1, single_trial_bin, cell, **kwargs)) == -1
        assert np.sign(self.log_likelihood(dist_name, b, single_trial_bin, cell, **kwargs) - self.log_likelihood(dist_name, b+0.1, single_trial_bin, cell, **kwargs)) == 1
        while b-a > 0.01:
            c = (a+b)/2
            if b-a>0.5:
                d = 0.1
            else:
                d = 0.01
            sign = np.sign(self.log_likelihood(dist_name, c, single_trial_bin, cell, **kwargs) - self.log_likelihood(dist_name, c+d, single_trial_bin, cell, **kwargs))
            if sign == 1:
                b = c
            else:
                a = c
        return a
    
    def KL(self, dist_name, k, recording, cell, n_samples=100, max_rate=300, **kwargs):
        
        dist_func = getattr(self, dist_name)

        if 'gaussian' in dist_name:
            r_list = np.linspace(-self.t, 2*self.t, n_samples)
        elif 'poisson' in dist_name:
            r_list = np.linspace(0, 2*self.t, n_samples)
        elif 'binomial' in dist_name:
            r_list = np.linspace(0, 1, n_samples)
            
        mean_list = [self.mean(dist_name, r, k, **kwargs) for r in r_list]
        if 'gaussian' in dist_name:
            a,b,c = curve_fit(self.sigmoid_para, r_list, mean_list)[0]
        else:
            f = interp1d(mean_list, r_list)
        
        kls = []
        mean_rate = recording.single_trial_bin.mean(0) * 100
        for rate in range(max_rate):
            if rate > self.t - 1:
                continue
            mean, _, em_dist, weight = recording.stats_rate(mean_rate, cell=cell, rate=rate)
            if np.isnan(mean):
                continue
            p_dist = em_dist[:self.t]
            if 'gaussian' in dist_name:
                r = self.inverse_sigmoid_para(rate/100, a, b, c)
            else:
                r = max(f(rate/100), 0)
            q_dist = np.array([dist_func(i, r, k, **kwargs) for i in range(self.t)])
            kl = np.sum([p_dist[i]*np.log(p_dist[i]/q_dist[i]) for i in range(self.t) if p_dist[i]>0])
            kls.append(kl)
        mean_kl = np.mean(kls) 
        
        return mean_kl
            
    
class recording_stats:
    
    def __init__(self, file_path, cells, filter_len=40):
        
        with h5py.File(file_path, 'r') as f:
            single_trial_bin = np.array(f['test']['repeats/binned'])
        self.single_trial_bin = np.swapaxes(single_trial_bin,1,2)[:, filter_len:, cells]
        self.n_cells = len(cells)
    
    def stats_rate(self, est_rates, cell, rate, intv=1):
        
        p = self.single_trial_bin[:, (est_rates[:,cell]>=rate-intv)*(est_rates[:,cell]<=rate+intv), cell].flatten()
        em_dist = np.array([(p == n).sum()/p.shape[0] for n in range(5)])
        weight = p.shape[0]
        return p.mean(), p.var(), em_dist, weight

    def smooth_single_trial(self, sigma):

        time_len = self.single_trial_bin.shape[1]
        time_upsample = np.linspace(0, (time_len-1)/100, time_len)
        response = []
        for cell in range(self.single_trial_bin.shape[2]):
            rate = estfr(self.single_trial_bin.mean(0)[:, cell], time_upsample, sigma=sigma)
            response.append(rate)
        smoothed_single_trial = np.stack(response).T

        return smoothed_single_trial

    def smooth_residual(self, sigma, cell, max_rate=300):
        smoothed_single_trial = self.smooth_single_trial(sigma)
        means = []
        variances = []
        rates = []
        for rate in range(max_rate):
            mean, _, _ = self.stats_rate(smoothed_single_trial, cell=cell, rate=rate)
            means.append(mean)
            rates.append(rate/100)
        residual = np.polyfit(np.array(rates)[~np.isnan(means)], np.array(means)[~np.isnan(means)], 3, full=True)[1]
        return residual