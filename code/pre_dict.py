import os
import pickle
import argparse
from collections import defaultdict
from utility import load_pickle, save_pickle, encode_sense_to_id, dict_to_id
# from pre_tf import create_infinity_mask

parser = argparse.ArgumentParser(description='Dataset preparation')

parser.add_argument('pickle_dir',
                    help='pickle directory')



def _word_to_id(string_set):
    """
    Create string to id dictionary.

    params:
        string_set (set)

    returns: dict
    """
    string_list = list(string_set) 
    word_to_id  = dict()
    word_to_id["<PAD>"] = 0
    word_to_id["<UNK>"] = 1 #OOV are mapped as <UNK>
    word_to_id.update({w:i+len(word_to_id) for i, w in enumerate(string_list)})
    return word_to_id

def _input_to_id(string_set):
    """
    Create string to id dictionary.

    params:
        string_set (set)

    returns: dict
    """
    string_list = list(string_set) 
    string_list.sort()

    word_to_id  = dict()
    word_to_id["<PAD>"] = 0
    word_to_id["<UNK>"] = 1 #OOV are mapped as <UNK>
    word_to_id.update({w:i+len(word_to_id) for i, w in enumerate(string_list)})
    return word_to_id


def _sense_to_id(senses_set):
    """
    Create label to id dictionary - All words without senses mapped to the same index.

    params:
        senses_set (set)

    returns: dict
    """
    senses = list(senses_set) 
    senses.sort() 

    label_to_id = dict()
    label_to_id["<PAD>"] = 0
    label_to_id["<UNK>"] = 1 #OOV are mapped as <UNK>
    label_to_id["<WRD>"] = 2 #Words are mapped as <WRD>
    label_to_id.update({w:i+len(label_to_id) for i, w in enumerate(senses)})
    return label_to_id


def _id_to_word(word_to_id):
    """
    Create id to string dictionary.

    params:
        word_to_id (dict)

    returns: dict
    """
    id_to_word = {v:k for k,v in word_to_id.items()}
    return id_to_word


def _value_to_id(sense_dict, val_dict):
	"""
	Prepares dict of possible senses as indices according to the dictionaries provided.

	params:
		sense_dict (dict)
		val_dict (dict)
	returns: dict
	"""
	index_dict  = dict()

	for key in sense_dict:
		index_list = list()
		for sense in sense_dict[key]:
			if sense in val_dict:   
				index_list.append(val_dict[sense])

		index_dict[key] = index_list

	return index_dict


def _key_to_id(sense_dict, key_dict):
	"""
	Prepares dict of possible senses as indices according to the dictionaries provided.

	params:
		sense_dict (dict)
		key_dict (dict)
	returns: dict
	"""
	index_dict  = dict()

	for key in sense_dict:
		index_dict[key_dict[key]] = sense_dict[key]

	return index_dict



def run(pickle_dir):
	"""
	Prepare data and dictionaries.

	params:
	    pickle_dir (str): Pickles directory path
	"""


	# -------------------------------------------------------------------- Start - Prepare all Dicts --------------------------------------------------------------------
	# #Possible Inputs
	# token_pos_set         = load_pickle(pickle_dir+"/sets/tokens_pos_set.pickle")
	# lemma_pos_set         = load_pickle(pickle_dir+"/sets/lemma_pos_set.pickle")

	# #Possible Outputs
	# fine_senses_set       = load_pickle(pickle_dir+"/sets/fine_senses_set.pickle")
	# coarse_domain_set     = load_pickle(pickle_dir+"/sets/coarse_domain_set.pickle")
	# coarse_lex_set        = load_pickle(pickle_dir+"/sets/coarse_lex_set.pickle")
	# pos_set               = load_pickle(pickle_dir+"/sets/pos_set.pickle")

	# #Create dictionaries
	# token_pos_dict        = _input_to_id(token_pos_set)
	# lemma_pos_dict        = _input_to_id(lemma_pos_set)

	# fine_senses_dict      = _sense_to_id(fine_senses_set)
	# coarse_domain_dict    = _sense_to_id(coarse_domain_set)
	# coarse_lex_dict       = _sense_to_id(coarse_lex_set)
	# pos_dict              = _sense_to_id(pos_set)

	# #Create inverted dictionaries
	# id_to_token_pos       = _id_to_word(token_pos_dict)
	# id_to_lemma_pos       = _id_to_word(lemma_pos_dict)

	# id_to_fine_senses     = _id_to_word(fine_senses_dict)
	# id_to_coarse_domain   = _id_to_word(coarse_domain_dict)
	# id_to_coarse_lex      = _id_to_word(coarse_lex_dict)
	# id_to_pos             = _id_to_word(pos_dict)


	# if not os.path.exists(pickle_dir+"/id_dicts"):
	#     os.makedirs(pickle_dir+"/id_dicts")

	# save_pickle(token_pos_dict, pickle_dir+"/id_dicts/token_pos_id_dict.pickle")
	# save_pickle(lemma_pos_dict, pickle_dir+"/id_dicts/lemma_pos_id_dict.pickle")

	# save_pickle(fine_senses_dict, pickle_dir+"/id_dicts/fine_senses_id_dict.pickle")
	# save_pickle(coarse_domain_dict, pickle_dir+"/id_dicts/coarse_domain_id_dict.pickle")
	# save_pickle(coarse_lex_dict, pickle_dir+"/id_dicts/coarse_lex_id_dict.pickle")
	# save_pickle(pos_dict, pickle_dir+"/id_dicts/pos_id_dict.pickle") 

	# save_pickle(id_to_token_pos, pickle_dir+"/id_dicts/id_to_token_pos.pickle")
	# save_pickle(id_to_lemma_pos, pickle_dir+"/id_dicts/id_to_lemma_pos.pickle")

	# save_pickle(id_to_fine_senses, pickle_dir+"/id_dicts/id_to_fine_senses.pickle")
	# save_pickle(id_to_coarse_domain, pickle_dir+"/id_dicts/id_to_coarse_domain.pickle")
	# save_pickle(id_to_coarse_lex, pickle_dir+"/id_dicts/id_to_coarse_lex.pickle")
	# save_pickle(id_to_pos, pickle_dir+"/id_dicts/id_to_pos.pickle")




	# print("token_pos_dict len", len(token_pos_dict))
	# # print("sen", senses_set)
	# print("lemma_pos_dict len", len(lemma_pos_dict))
	# # print("no_sen", no_senses)
	# print("fine_senses_dict len", len(fine_senses_dict))

	# print("coarse_domain_dict len", len(coarse_domain_dict))

	# print("coarse_lex_dict len", len(coarse_lex_dict))
	# # print("sen", senses_set)
	# print("pos_dict len", len(pos_dict))
	# # print("no_sen", no_senses)

	# print("id_to_token_pos len", len(id_to_token_pos))
	# # print("sen", senses_set)
	# print("id_to_lemma_pos len", len(id_to_lemma_pos))
	# # print("no_sen", no_senses)

	# print("id_to_fine_senses len", len(id_to_fine_senses))

	# print("id_to_coarse_domain len", len(id_to_coarse_domain))
	# # print("sen", senses_set)
	# print("id_to_coarse_lex len", len(id_to_coarse_lex))
	# # print("no_sen", no_senses)
	# print(" id_to_pos len", len(id_to_pos))
	# # print("lab", labels_set)

	# print("token_pos_dict", token_pos_dict)


	# print("fine_senses_dict", fine_senses_dict)
	# print("coarse_domain_dict", coarse_domain_dict)
	# print("coarse_lex_set", coarse_lex_set)
	# lemma_pos_id_dict           = load_pickle(pickle_dir+"/id_dicts/lemma_pos_id_dict.pickle")
	# id_to_lemma_pos           = load_pickle(pickle_dir+"/id_dicts/id_to_lemma_pos.pickle")

	# print("lemma_pos_id_dict", lemma_pos_id_dict)
	# print("id_to_lemma_pos", id_to_lemma_pos)


	# print("id_to_fine_senses", id_to_fine_senses)
	# print("id_to_coarse_domain", id_to_coarse_domain)
	# print("id_to_coarse_lex", id_to_coarse_lex)


	# -------------------------------------------------------------------- End - Prepare all Dicts --------------------------------------------------------------------




    # max_length = 60

 #    input_tokens           = load_pickle(pickle_dir+"/input_tokens_list.pickle")
 #    input_lemma_pos        = load_pickle(pickle_dir+"/input_lemma_pos_list.pickle")
 #    output_labels          = load_pickle(pickle_dir+"/output_labels_list.pickle")

 #    index_input_tokens     = encode_to_id(input_tokens, words_dict, max_length)
 #    index_input_lemma_pos  = encode_to_id(input_lemma_pos, lemma_pos_dict, max_length)
   
 #    index_output_labels    = encode_to_id(output_labels, labels_dict, max_length)
 #    index_output_senses    = encode_sense_to_id(output_labels, senses_dict, words_set, max_length)

 #    save_pickle(index_input_tokens, pickle_dir+"/index_input_tokens.pickle")
 #    save_pickle(index_input_lemma_pos, pickle_dir+"/index_input_lemma_pos.pickle")
  
 #    save_pickle(index_output_labels, pickle_dir+"/index_output_labels.pickle")
 #    save_pickle(index_output_senses, pickle_dir+"/index_output_senses.pickle")



	# #Convert senses dicts to index labels dicts         
	# -------------------------------------------------------------------- Convert candidate dictionaries to indices --------------------------------------------------------------------
	# lemma_pos_id_dict              = load_pickle(pickle_dir+"/id_dicts/lemma_pos_id_dict.pickle")
	# fine_senses_id_dict            = load_pickle(pickle_dir+"/id_dicts/fine_senses_id_dict.pickle")
	# coarse_domain_id_dict          = load_pickle(pickle_dir+"/id_dicts/coarse_domain_id_dict.pickle")
	# coarse_lex_id_dict             = load_pickle(pickle_dir+"/id_dicts/coarse_lex_id_dict.pickle")
	# pos_id_dict               	 = load_pickle(pickle_dir+"/id_dicts/pos_id_dict.pickle")

	# cand_fine_senses               = load_pickle(pickle_dir+"/cand_dicts/cand_fine_senses.pickle")
	# cand_coarse_domain             = load_pickle(pickle_dir+"/cand_dicts/cand_coarse_domain.pickle")
	# cand_coarse_lex                = load_pickle(pickle_dir+"/cand_dicts/cand_coarse_lex.pickle")
	# cand_pos               		 = load_pickle(pickle_dir+"/cand_dicts/cand_pos.pickle")

	# print("cand_fine_senses", cand_fine_senses)
	# print("cand_coarse_domain", cand_coarse_domain)
	# print("cand_coarse_lex", cand_coarse_lex)
	# print("cand_pos", cand_pos)

	# id_cand_fine_senses            = _value_to_id(cand_fine_senses, fine_senses_id_dict)
	# id_cand_coarse_domain          = _value_to_id(cand_coarse_domain, coarse_domain_id_dict)
	# id_cand_coarse_lex             = _value_to_id(cand_coarse_lex, coarse_lex_id_dict)
	# id_cand_pos                    = _value_to_id(cand_pos, pos_id_dict)

	# print("id_cand_fine_senses", id_cand_fine_senses)
	# print("id_cand_coarse_domain", id_cand_coarse_domain)
	# print("id_cand_coarse_lex", id_cand_coarse_lex)

	# save_pickle(id_cand_fine_senses, pickle_dir+"/cand_dicts/id_cand_fine_senses.pickle")
	# save_pickle(id_cand_coarse_domain, pickle_dir+"/cand_dicts/id_cand_coarse_domain.pickle")
	# save_pickle(id_cand_coarse_lex, pickle_dir+"/cand_dicts/id_cand_coarse_lex.pickle")
	# save_pickle(id_cand_pos, pickle_dir+"/cand_dicts/id_cand_pos.pickle")

	# all_id_cand_fine_senses            = _key_to_id(id_cand_fine_senses, lemma_pos_id_dict)
	# all_id_cand_coarse_domain          = _key_to_id(id_cand_coarse_domain, lemma_pos_id_dict)
	# all_id_cand_coarse_lex             = _key_to_id(id_cand_coarse_lex, lemma_pos_id_dict)
	# all_id_cand_pos                    = _key_to_id(id_cand_pos, lemma_pos_id_dict)

	# print("id_cand_fine_senses", id_cand_fine_senses)
	# print("id_cand_coarse_domain", id_cand_coarse_domain)
	# print("id_cand_coarse_lex", id_cand_coarse_lex)

	# save_pickle(all_id_cand_fine_senses, pickle_dir+"/cand_dicts/all_id_cand_fine_senses.pickle")
	# save_pickle(all_id_cand_coarse_domain, pickle_dir+"/cand_dicts/all_id_cand_coarse_domain.pickle")
	# save_pickle(all_id_cand_coarse_lex, pickle_dir+"/cand_dicts/all_id_cand_coarse_lex.pickle")
	# save_pickle(all_id_cand_pos, pickle_dir+"/cand_dicts/all_id_cand_pos.pickle")

	# print("all_id_cand_fine_senses", all_id_cand_fine_senses)
	# print("all_id_cand_coarse_domain", all_id_cand_coarse_domain)
	# print("all_id_cand_coarse_lex", all_id_cand_coarse_lex)
	# print("all_id_cand_pos", all_id_cand_pos)
	# -------------------------------------------------------------------- End - Convert candidate dictionaries to indices --------------------------------------------------------------------

	



if __name__ == '__main__':
    args = parser.parse_args()
    run(**vars(args))
