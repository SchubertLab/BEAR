import nltk
import re
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
import pandas as pd
from time import time
from collections import OrderedDict
import argparse
import os
import platform


parser = argparse.ArgumentParser()
parser.add_argument('--lags', type=int, default=5)
parser.add_argument('--dataset', type=str, default='uniref90_humans')
args = parser.parse_args()

lags = args.lags
dataset = args.dataset
amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'X', 'Y', 'V', ']']
amino_acid_idx = OrderedDict({k: v for v, k in enumerate(amino_acids)})

fdists = OrderedDict()
fdists_plus_1 = OrderedDict()
df_transition_matrices = OrderedDict()
sets = ['train_val', 'test']

for lag in tqdm(range(1, lags)):
	for set in sets:
		t = time()
		data = list(SeqIO.parse(f'data/{dataset}_{set}.fasta', 'fasta'))
		data = [re.sub('[UZOB]', 'X', str(rec.seq)[:512]) for rec in data]
		data = ['['*lag + rec + ']'*(lag+1) for rec in data]
		data = ''.join(data)

		ngrams = nltk.ngrams(data, lag)
		ngrams_plus_1 = nltk.ngrams(data, lag+1)
		fdist = nltk.FreqDist(ngrams)
		fdist = {''.join(k): v for k, v in fdist.items()}
		fdist = OrderedDict({k: v for k, v in fdist.items() if ']' not in k})
		fdist_plus_1 = nltk.FreqDist(ngrams_plus_1)
		fdist_plus_1 = OrderedDict({''.join(k): v for k, v in fdist_plus_1.items()})

		fdists[set] = fdist
		fdists_plus_1[set] = fdist_plus_1

		print(f'\n{set} N-Gram lag {lag} {(time() - t):.2} s')
		t = time()

		transition_matrix = np.zeros([len(fdist), len(amino_acids)], dtype=int)

		for i, (ngram, v) in enumerate(fdist.items()):
			for next_aa in amino_acids:
				if ngram+next_aa in fdist_plus_1:
					transition_matrix[i, amino_acid_idx[next_aa]] = fdist_plus_1[ngram+next_aa]

		df_transition_matrix = pd.DataFrame(transition_matrix, index=fdist.keys(), columns=amino_acids)
		df_transition_matrices[set] = df_transition_matrix
		print(f'{set} Transition Matrix lag {lag} {(time() - t):.2} s')

	result = pd.concat(df_transition_matrices.values(), axis=1, names=df_transition_matrices.keys())
	result = result.fillna(0)
	transition_matrices = result.to_numpy(dtype=int)
	transition_matrices = np.array([transition_matrices[:, i*len(amino_acids):(i+1)*len(amino_acids)] for i in range(len(sets))]).swapaxes(0, 1).tolist()

	# Reset
	f = open(f'data/{dataset}_transition_count_lag_{lag}.tsv', 'w')
	f.write('')

	# Write
	f = open(f'data/{dataset}_transition_count_lag_{lag}.tsv', 'a')
	ngrams_unified = result.index.tolist()
	for ngram_, trans_mat in zip(ngrams_unified, transition_matrices):
		f.write(ngram_ + '\t' + str(trans_mat) + '\n')
	f.close()
	print(os.getcwd())
	if 'bear_model' not in os.getcwd():
		os.chdir('bear_model')
	if platform.system() == 'Linux':
		os.system(f'shuf data/{dataset}_transition_count_lag_{lag}.tsv -o data/{dataset}_transition_count_lag_{lag}_shuf.tsv')
	elif platform.system() == 'Darwin':
		os.system(f'gshuf data/{dataset}_transition_count_lag_{lag}.tsv -o data/{dataset}_transition_count_lag_{lag}_shuf.tsv')
