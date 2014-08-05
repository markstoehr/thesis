from __future__ import division
import numpy as np
from scipy import linalg

def make_fft_templates(mean,cov,n_fft_times):
    """
    Returns template_mean_cov_fft, template_cov_fft, normal_constant
    which are the workhorses for doing detections
    
    Assumptions is that the fastest dimension is with respect to
    time (along which the convolutions are being performed
    """
    cov_inv = cov**-1
    n_template_entries = np.prod(mean.shape[-2:])
    mean_cov_inv = mean*cov_inv
    template_mean_cov_fft = np.fft.fft(mean_cov_inv,n=n_fft_times,axis=-1).conj()
    template_cov_fft = np.fft.fft(cov_inv,n=n_fft_times,axis=-1).conj()
    normal_constant = -.5*( n_template_entries*np.log(2*np.pi) + np.log(cov).sum(-1).sum(-1) + (mean_cov_inv * mean).sum(-1).sum(-1))

    return template_mean_cov_fft, template_cov_fft, normal_constant

def prep_utterance_overlap_add(utt,n_fft_times,n_template_times):
    """
    Input is an utterance utt,
    number of fft tiems, number of template times in the templates
    output is 

    utt_fft_chunks
    utt_sq_fft_chunks
    """
    n_features,n_times = utt.shape
    utt_chunk_length = n_fft_times - n_template_times + 1
    full_n_times = n_times + 2* (n_template_times -1)
    n_utt_chunks = int(np.ceil( full_n_times / utt_chunk_length)+.01)
    utt_fft_chunks = np.zeros((n_utt_chunks, n_features, n_fft_times),dtype=np.complex128)
    utt_sq_fft_chunks = np.zeros((n_utt_chunks, n_features, n_fft_times),dtype=np.complex128)
    for i in xrange(n_utt_chunks):
            start_idx = utt_chunk_length*i
            end_idx = min(utt_chunk_length*(i+1),n_times)
            utt_fft_chunks[i] = np.fft.fft(utt[:,start_idx:end_idx],n=n_fft_times)
            utt_sq_fft_chunks[i] = np.fft.fft(utt[:,start_idx:end_idx]**2,n=n_fft_times)

    return utt_fft_chunks, utt_sq_fft_chunks

def perform_overlap_add_detection(utt_fft_chunks, utt_sq_fft_chunks,template_mean_cov_fft, template_cov_fft, normal_constant,n_template_times):
    """
    Get the overlap add detector outputs
    """
    n_utt_chunks = utt_fft_chunks.shape[0]
    n_fft_times = template_cov_fft.shape[-1]
    n_overlap_likelihoods = (n_utt_chunks-1) * (n_fft_times - n_template_times+1) + n_fft_times
    fast_mult_overlap_chunk_response = np.real(np.fft.ifft((template_mean_cov_fft*utt_fft_chunks - .5 * template_cov_fft * utt_sq_fft_chunks).sum(-2) * np.exp(2j*np.pi * np.arange(n_fft_times)/n_fft_times * -(n_template_times - 1)),axis=-1))
    overlap_add_likelihoods = normal_constant * np.ones(n_overlap_likelihoods)
    for i in xrange(n_utt_chunks):
        overlap_start_idx = i*(n_fft_times - n_template_times + 1)
        overlap_add_likelihoods[overlap_start_idx:overlap_start_idx+n_fft_times] += fast_mult_overlap_chunk_response[i]

    return overlap_add_likelihoods


def phn_frame_bounds_from_sample_bounds(start_end_samples,frame_inds):
    """
    Code infers the boundaries with respect to frames
    from the boundaries with respect to samples

    Takes in the phn file and the frame indices to find it

    And frame_inds which indicates the indices containing
    the 
    """
    phn_sample_boundaries = start_end_samples[:,1].astype(int)
    phn_frame_boundaries = []
    for phn_sample_boundary in phn_sample_boundaries:
        best_match_id = 0
        best_match_score = -10
        for frame_id, frame_ind in enumerate(frame_inds):
            num_above = (frame_ind >= phn_sample_boundary).sum()
            num_below = (frame_ind <= phn_sample_boundary).sum()
            cur_score = num_above * num_below
            if cur_score > best_match_score:
                best_match_id = frame_id
                best_match_score = cur_score
                best_num_above = num_above
                best_num_below = num_below
                
        if best_num_below < best_num_above:
            phn_frame_boundaries.append(best_match_id)
        else:
            phn_frame_boundaries.append(best_match_id+1)

    phn_frame_transcript = [(0,phn_frame_boundaries[0])]
    for p_id in xrange(1,len(phn_frame_boundaries)):
        phn_frame_transcript.append(
            (phn_frame_boundaries[p_id-1],
             phn_frame_boundaries[p_id]))
    
            
    return np.array(phn_frame_transcript,dtype=int)

def create_phn_map(map_fl_lines):
    map_fl_lines = [f.strip().split() for f in map_fl_lines]
    map_dict = {}
    for fl_line in map_fl_lines:
        if len(fl_line) == 2:
            map_dict[fl_line[0]] = fl_line[1]

    return map_dict
    

def map_utterance_transcript(phn_file_lines,map_dict):
    start_end_samples = []
    transcript_phns = []
    for l in phn_file_lines:
        lsplit = l.strip().split()
        if len(lsplit) == 3:
            start_sample = int(lsplit[0])
            end_sample = int(lsplit[1])
            cur_phn = map_dict[lsplit[2]] if map_dict.has_key(lsplit[2]) else ''
            start_end_samples.append((start_sample,end_sample))
            
            transcript_phns.append(cur_phn)

    return np.array(start_end_samples,dtype=int),transcript_phns
