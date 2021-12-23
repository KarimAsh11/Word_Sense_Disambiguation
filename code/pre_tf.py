import numpy as np
from collections import defaultdict
from parse_utility import get_map_dict


def calc_f1_mfs(pred, gold):
	"""
	Calculates F1 score - gold and pred files for final evaluation of the modele.

	params:
		pred (str)
		gold (str)
	returns: float
	"""	
	pred_map = get_map_dict(pred)
	gold_map = get_map_dict(gold)
	
	F1 = 0

	for x in pred_map:
		if pred_map[x] == gold_map[x]:
			F1+=1

	return F1/len(pred_map)

# print(calc_f1_mfs("../resources/final_2007_fine.txt", "../resources/gold_fine_2007.txt"))


def calc_f1(pred, labels, sense_dict):
	"""
	Calculates F1 score - arrays.

	params:
		pred       (numpy)
		labels     (numpy)
		sense_dict (dict)
	returns: float
	"""	
	TS         = 0
	TPR        = 0
	CC         = 0
	no_pred    = [0,1,2]

	for i in range(pred.shape[0]):

		if pred[i] not in no_pred:
			TPR+=1
		if labels[i] not in no_pred:
			TS+=1
		if pred[i] == labels[i] and labels[i] not in no_pred:
			CC+=1

	precision  = CC/(TPR+1e-6)
	recall     = CC/(TS+1e-6)

	F1 = (2*precision*recall) / (precision+recall+ 1e-5)
	return F1


def clean_id_dict(dictionary):
	"""
	Prepares id to string dict for senses only.

	params:
		dictionary (dict)
	returns: 
	"""	
	sense_id_dict = dict()
	for key in dictionary:
		if "wn:" in dictionary[key]:
			sense_id_dict[key] = dictionary[key]
	return sense_id_dict


def prepare_mask_infinity(file, inst_mask, inf_dict, max_length, class_num):
	"""
	Prepares the inifinity mask.

	params:
		file       (numpy array)
		inst_mask  (numpy array)
		inf_dict   (dict)
		max_length (int)
		class_num  (int)
	returns: numpy array
	"""
	k = 0
	number_of_ex = file.shape[0]
	inf_mask     = np.zeros([number_of_ex, max_length, class_num])

	for i, line in enumerate(file):

		# ninf_unk = np.ones((class_num)) * np.finfo(np.float32).min
		ninf_wrd = np.ones((class_num)) * (-1e+15)
		ninf_wrd[2] = 0
		ninf_unk = np.ones((class_num)) * (-1e+15)
		ninf_unk[1] = 0
		ninf_pad = np.ones((class_num)) * (-1e+15)
		ninf_pad[0] = 0

		for j, word in enumerate(line):
			if inst_mask[i][j]  == 0:

				if word == 0:
					inf_mask[i,j] = ninf_pad
						
				elif word == 1:
					inf_mask[i,j] = ninf_unk
			
				else:
					inf_mask[i,j] = ninf_wrd
			
			else:
				if word in inf_dict:
					inf_mask[i,j] = get_ninf_syns(word, inf_dict, class_num)
				
				else:
					# print("word:", word)
					inf_mask[i,j] = ninf_unk

	return inf_mask


def infinity_dict(vocab_dict, dictionary):
	"""
	Prepares the infinity dictionary according to the required set of words.

	params:
		vocab_dict (dict)
		dictionary (dict)
	returns: 
	"""	
	vocab = list(vocab_dict.keys())
	inf_dict = defaultdict(set)
	inf_dict = dictionary
	for v in vocab:
		if v not in dictionary:
			inf_dict[v].add(v.rsplit("_",1)[0])
	return inf_dict


def inf_dict_ID(inf_dict, key_dict, val_dict):
	"""
	Prepares inf dict as indices according to the dictionaries provided.

	params:
		inf_dict (dict)
		key_dict (dict)
		val_dict (dict)
	returns: dict
	"""
	inf_dict_ID  = defaultdict(set)
	un = 0
	kn = 0
	for key in key_dict:
		index_key = key_dict[key]
		for sense in inf_dict[key]:
			kn += 1
			if sense in val_dict:   
				inf_dict_ID[index_key].add(val_dict[sense])
			else:
				inf_dict_ID[index_key].add(val_dict["<UNK>"])
				un+=1

	return inf_dict_ID


def inf_dict_ID_WRD(inf_dict, key_dict, val_dict, vocab):
	"""
	Prepares inf dict as indices according to the dictionaries provided - all words without possible senses mapped to <WRD>.

	params:
		inf_dict (dict)
		key_dict (dict)
		val_dict (dict)
		vocab    (set)
	returns: dict
	"""
	inf_dict_ID  = defaultdict(set)

	for key in key_dict:
		index_key = key_dict[key]
		for sense in inf_dict[key]:
			if sense in val_dict:   
				inf_dict_ID[index_key].add(val_dict[sense])
			elif sense in vocab:
				inf_dict_ID[index_key].add(val_dict["<WRD>"])
			else:
				inf_dict_ID[index_key].add(val_dict["<UNK>"])

	return inf_dict_ID


def get_ninf_syns(word, inf_dict, class_num):
	"""
	Prepares the infinity mask for the possible senses of the given word.

	params:
		word       (int)
		inf_mask   (dict)
		class_num  (int)
	returns: numpy array
	"""
	# ninf_word = np.ones((class_num)) * np.finfo(np.float32).min
	ninf_word = np.ones((class_num)) * (-1e+15)
	if word in inf_dict:
		for v in inf_dict[word]:
			ninf_word[v] = 0
	else:
		ninf_word[2] = 0 
	return ninf_word


def batch_generator_base(X, Y, inf_dict, batch_size, shuffle=False):
	if not shuffle:
		for start in range(0, len(X1), batch_size):
			end = start + batch_size
			inf_fine_mask = prepare_mask_infinity(X[start:end], inf_dict, max_length, class_num)
			yield X[start:end], Y[start:end], inf_fine_mask
	else:
		perm = np.random.permutation(len(X))
		for start in range(0, len(X1), batch_size):
			end = start + batch_size
			inf_fine_mask = prepare_mask_infinity(X[perm[start:end]], inf_dict, max_length, class_num)
			yield X[perm[start:end]], Y[perm[start:end]], inf_fine_mask







