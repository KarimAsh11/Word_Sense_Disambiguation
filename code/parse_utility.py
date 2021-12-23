import time
from nltk.corpus import wordnet as wn


def get_map_dict(file_path):
	"""
	Get map.

	params:
	    file_path (str)

	returns: dict
	"""
	with open (file_path, 'r') as f:
		map_dict = dict()
		for row in f:
			sp  = row.split()
			map_dict[sp[0]]  = sp[1]
		return map_dict


def get_map_dict_tsv(file_path):
	"""
	Get map from tsv file.

	params:
	    file_path (str)

	returns: dict
	"""
	with open (file_path, 'r') as f:
		map_dict = dict()
		for row in f:
			sp  = row.split()
			map_dict[sp[0]]  = sp[1]
		return map_dict


def get_map_dict_tsv_inv(file_path):
	"""
	Get map from tsv file - inverted.

	params:
	    file_path (str)

	returns: dict
	"""
	with open (file_path, 'r') as f:
		map_dict = dict()
		for row in f:
			sp  = row.split()
			map_dict[sp[1]]  = sp[0]
		return map_dict


def get_syns(sense):
	"""
	Retrieve WordNet synset.

	params:
	    sense (str)

	returns: str
	"""
	synset = wn.synset_from_sense_key(sense)
	synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
	# print("syns_id", synset_id)	

	return synset_id


def get_wn_syns(sense):
	"""
	Retrieve WordNet synset.

	params:
	    sense (str)

	returns: str
	"""
	synset = wn.lemma_from_key(sense).synset()
	synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()

	return synset_id


