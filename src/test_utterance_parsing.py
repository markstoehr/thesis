from __future__ import division
import numpy as np
from scipy import linalg

# create synthetic utterance
n_times = 100
n_features = 3

np.random.seed(0)
utt = np.random.randn(n_features,n_times)
utt_sq = utt**2

n_template_times = 5
n_template_entries = n_features * n_template_times

template_mean = np.random.randn(n_features,n_template_times)
template_cov = np.maximum(np.random.randn(n_features,n_template_times)**2,np.random.randn(n_features,n_template_times)**2)**-1

template_mean_cov = template_mean *template_cov

normal_constant = -.5*( n_template_entries*np.log(2*np.pi) - np.log(template_cov).sum() + np.sum(template_mean_cov * template_mean))

n_likelihoods = n_times - n_template_times +1
base_likelihoods = np.zeros(n_likelihoods)

for t in xrange(n_likelihoods):
    base_likelihoods[t] = np.sum(utt[:,t:t+n_template_times]* template_mean_cov) - .5* np.sum(utt_sq[:,t:t+n_template_times]*template_cov) + normal_constant


n_fft_times = 32

template_mean_cov_fft = np.fft.fft(template_mean_cov,n=n_fft_times,axis=-1).conj()
template_cov_fft = np.fft.fft(template_cov,n=n_fft_times,axis=-1).conj()

# chop the utterance into chunks that we can then do the overlap add algorithm on
# we need
# n_utt_chunks * (n_fft_times - n_template_times + 1) >= n_times
utt_chunk_length = n_fft_times - n_template_times + 1
n_utt_chunks = int(np.ceil( n_times / utt_chunk_length))
utt_fft_chunks = np.zeros((n_utt_chunks, n_features, n_fft_times),dtype=np.complex128)
utt_sq_fft_chunks = np.zeros((n_utt_chunks, n_features, n_fft_times),dtype=np.complex128)
n_overlap_likelihoods = (n_utt_chunks-1) * (n_fft_times - n_template_times+1) + n_fft_times
overlap_add_likelihoods = np.zeros(n_overlap_likelihoods)
for i in xrange(n_utt_chunks):
    start_idx = utt_chunk_length*i
    end_idx = min(utt_chunk_length*(i+1),n_times)
    utt_fft_chunks[i] = np.fft.fft(utt[:,start_idx:end_idx],n=n_fft_times)
    utt_sq_fft_chunks[i] = np.fft.fft(utt_sq[:,start_idx:end_idx],n=n_fft_times)
    overlap_start_idx = i*(n_fft_times - n_template_times + 1)
    overlap_add_likelihoods[overlap_start_idx:overlap_start_idx + n_fft_times] += np.real(np.fft.ifft((template_mean_cov_fft*utt_fft_chunks[i] - .5*template_cov_fft * utt_sq_fft_chunks[i]).sum(0) * np.exp(2j*np.pi * np.arange(n_fft_times)/n_fft_times*-(n_template_times -1)),axis=-1))
    
overlap_add_likelihoods += normal_constant
np.testing.assert_array_almost_equal(overlap_add_likelihoods[4:100], base_likelihoods)


np.real(np.fft.ifft((template_mean_cov_fft*utt_fft_chunks[0] - .5*template_cov_fft * utt_sq_fft_chunks[0]).sum(0),axis=-1)) + normal_constant

likelihoods_fft = np.real(np.fft.ifft((template_mean_cov_fft * utt_fft -.5* template_cov_fft * utt_sq_fft).sum(0),axis=-1)) + normal_constant




v = np.real(np.fft.ifft((template_mean_cov_fft * utt_fft).sum(0)))[:10]
v2 = np.real(np.fft.ifft((template_cov_fft * utt_sq_fft).sum(0)))[:10]
v3 = np.real(np.fft.ifft((template_mean_cov_fft * utt_fft - .5 * template_cov_fft * utt_sq_fft).sum(0)))[:10]
for i in xrange(10):
    print np.sum(template_mean_cov * utt[:,i:i+5]) - v[i]
    print np.sum(template_cov * utt_sq[:,i:i+5]) - v2[i]
    print np.sum(template_mean_cov * utt[:,i:i+5] - .5*(template_cov * utt_sq[:,i:i+5])) - v3[i]




u = np.random.randn(100)
x = np.random.randn(10)
x_ex = np.zeros(100)
x_ex[:10] = x

C = linalg.circulant(x_ex).T
y = np.dot(C,u)

y2 = np.zeros(100)
for i in xrange(100-10+1):
    y2[i] = np.dot(x,u[i:i+10])


F = np.fft.fft(np.eye(100))

np.real(np.dot(F.T.conj(),np.dot(np.diag(np.fft.fft(C[0])),F)))/100 - C

y3 = np.real(np.fft.ifft(np.fft.fft(x_ex).conj() * np.fft.fft(u)))
np.abs(y3 - y2)
