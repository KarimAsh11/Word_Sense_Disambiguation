import pickle
import numpy as np
from collections import defaultdict

def write(content, file_path, encoding='utf8'):
    """
    Write to output file

    params:
        content List[List[str]]
        file_path (str)
        encoding (str) [default='utf8']
    """
    with open(file_path, 'w', encoding=encoding) as f:
        for line in content:
            f.write(''.join(line))
            if line != content[-1]:
                f.write('\n')


def write_str(content, file_path, encoding='utf8'):
    """
    Write to output file

    params:
        content List[List[str]]
        file_path (str)
        encoding (str) [default='utf8']
    """
    with open(file_path, 'w', encoding=encoding) as f:
        for line in content:
            f.write(''.join(str(line)))
            if line != content[-1]:
                f.write('\n')



def save_pickle(file, pickle_file):
    """
    Pickle File.

    params:
        file
        pickle_file (str)
    """
    with open(pickle_file, 'wb') as h:
        pickle.dump(file, h, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(pickle_file):
	"""
	load pickle.

	params:
	    pickle_file (pickle)
	"""
	with open(pickle_file, 'rb') as h:
		return pickle.load(h)


def clean_vec(vec_path, clean_vec_path):
	"""
	Keeps sense embeddings only.

	params:
	    vec_path       (str)
	    clean_vec_path (str)
	"""
	vectors    = read(vec_path).splitlines()
	sense_vec  = []

	_, emb_size   = vectors[0].split()

	for vec in vectors:
		if '_bn:' in vec:
			sense_vec.append(vec)

	first_line = str(len(sense_vec)) + " " + emb_size
	sense_vec.insert(0, first_line)


	write(sense_vec, clean_vec_path)


def encode_elmo(file, max_length):
	"""
	Prepares input for elmo.

	params:
		file (list)
		max_length (int)
	returns: list
	"""
	encoded = []
	for line in file: 
		if  len(line)>max_length: 
			pad_line = line[:max_length] 
		else: 
			pad_line = line + [''] * (max_length - len(line)) 
		
		encoded.append(pad_line) 

	return np.asarray(encoded)


def encode_input_to_id(file, dictionary, max_length):
	"""
	Prepares numpy array of indices according to the dictionary.

	params:
		file (list)
		dictionary (dict)
		max_length (int)
	returns: numpy array
	"""
	number_of_ex = len(file)
	encoded      = np.zeros([number_of_ex, max_length])

	for i, line in enumerate(file):
		for j, word in enumerate(line):
			if j > max_length-1:
				break
			if word in dictionary:   
				encoded[i,j] = dictionary[word]
			else:
				encoded[i,j] = dictionary["<UNK>"]

	return encoded


def encode_sense_to_id(file, dictionary, vocab, max_length):
	"""
	Prepares numpy array of indices according to the dictionary - Mapping all words in vocab to <WRD>.

	params:
		file (list)
		dictionary (dict)
		vocab (list)
		max_length (int)
	returns: numpy array
	"""
	number_of_ex = len(file)
	encoded      = np.zeros([number_of_ex, max_length])

	for i, line in enumerate(file):
		for j, word in enumerate(line):
			if j > max_length-1:
				break
			if word in dictionary:   
				encoded[i,j] = dictionary[word]
			elif word in vocab:
				encoded[i,j] = dictionary["<WRD>"]
			else:
				encoded[i,j] = dictionary["<UNK>"]

	return encoded


def encode_instance_mask(file, max_length):
	"""
	Prepares instance mask as numpy array.

	params:
		file (list)
		max_length (int)
	returns: numpy array
	"""
	number_of_ex = len(file)
	encoded      = np.zeros([number_of_ex, max_length])

	for i, line in enumerate(file):
		for j, word in enumerate(line):
			if j > max_length-1:
				break
			if word == 0:   
				encoded[i,j] = 0
			else:
				encoded[i,j] = 1

	return encoded


def dict_to_id(sense_dict, key_dict, val_dict):
	"""
	Prepares dict of possible senses as indices according to the dictionaries provided.

	params:
		sense_dict (dict)
		key_dict (dict)
		val_dict (dict)
	returns: dict
	"""
	index_dict  = defaultdict(set)

	for key in sense_dict:
		index_key = key_dict[key]
		for sense in sense_dict[key]:
			if sense in val_dict:   
				index_dict[index_key].add(val_dict[sense])

	return index_dict


