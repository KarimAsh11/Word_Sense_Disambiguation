import nltk
import string
from collections import defaultdict
from lxml import etree
from argparse import ArgumentParser
from parse_utility import get_syns, get_wn_syns, get_map_dict, get_map_dict_tsv, get_map_dict_tsv_inv
from utility import *
import time


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()



def semcor_parse(input_path, out_path, resources_path):
	"""
	Parses SemCor dataset.

	params:
	    input_path		(str)
	    out_path		(str)
	    resources_path	(str)
	"""
	i=0
	corp_token               = []
	corp_token_pos           = []
	corp_lemma_pos           = []
	corp_fine_grain          = []
	corp_coarse_domain       = []
	corp_coarse_lex          = []
	corp_pos                 = []
	corp_instance            = []

	tokens_set               = set()
	tokens_pos_set           = set()
	lemma_pos_set            = set()
	fine_senses_set          = set()
	coarse_domain_set        = set()
	coarse_lex_set           = set()
	pos_set                  = set()
	wf_vocab_set             = set()
	
	cand_fine_senses         = defaultdict(set)
	cand_coarse_domain       = defaultdict(set)
	cand_coarse_lex          = defaultdict(set)
	cand_pos                 = defaultdict(set)

	fine_sense_path		     = resources_path+"/semcor.gold.key.txt"
	babel_word_path		     = resources_path+"/babelnet2wordnet.tsv"
	babel_dom_path		     = resources_path+"/babelnet2wndomains.tsv"
	babel_lex_path		     = resources_path+"/babelnet2lexnames.tsv"
	file_path		         = resources_path+"/"+input_path

	fine_sense_map           = get_map_dict(fine_sense_path)
	word_to_babel_map        = get_map_dict_tsv_inv(babel_word_path)
	babel_to_word_map        = get_map_dict_tsv(babel_word_path)
	babel_to_dom_map         = get_map_dict_tsv(babel_dom_path)
	babel_to_lex_map         = get_map_dict_tsv(babel_lex_path)

	print("Parsing XML file ...")
	for event, element in etree.iterparse(file_path, tag="sentence"):
		token_sentence          = []
		token_pos_sentence      = []
		lemma_pos_sentence      = []
		fine_grain_sentence     = []
		coarse_domain_sentence  = []
		coarse_lex_sentence     = []
		pos_sentence            = []
		instance_sentence       = []

		i+=1

		if event == "end":
			for el in element.iter():
				if el.text is not None:
					if el.tag == "wf":
						token = el.text.lower().replace(" ", "_")
						lemma = el.attrib["lemma"].lower().replace(" ", "_")
						pos   = el.attrib["pos"].lower()

						token_pos  = '_'.join([token, pos])
						lemma_pos = '_'.join([lemma, pos])

						tokens_set.add(token)
						tokens_pos_set.add(token_pos)
						lemma_pos_set.add(lemma_pos)
						pos_set.add(pos)
						wf_vocab_set.add(token)

						token_sentence.append(token)
						token_pos_sentence.append(token_pos)
						lemma_pos_sentence.append(lemma_pos)
						fine_grain_sentence.append(token)
						coarse_domain_sentence.append(token)
						coarse_lex_sentence.append(token)
						pos_sentence.append(pos)
						instance_sentence.append(0)
			
					if el.tag == "instance":
						token = el.text.lower().replace(" ", "_")
						lemma = el.attrib["lemma"].lower()
						pos   = el.attrib["pos"].lower()

						token_pos  = '_'.join([token, pos])
						lemma_pos = '_'.join([lemma, pos])

						if el.attrib['id']:
							wn_syns = get_wn_syns(fine_sense_map[el.attrib["id"]])
							if word_to_babel_map.get(wn_syns) is not None:
								bn_syns  = word_to_babel_map[wn_syns]
								fine_senses_set.add(bn_syns)
								cand_fine_senses[lemma_pos].add(bn_syns)

								if babel_to_dom_map.get(bn_syns) is not None:
									domain = babel_to_dom_map[bn_syns]
									coarse_domain_set.add(domain)
									cand_coarse_domain[lemma_pos].add(domain)
								else:
									domain = "factotum"
									coarse_domain_set.add(domain)
									cand_coarse_domain[lemma_pos].add(domain)
									
								if babel_to_lex_map.get(bn_syns) is not None:
									lex_name = babel_to_lex_map[bn_syns]
									coarse_lex_set.add(lex_name)
									cand_coarse_lex[lemma_pos].add(lex_name)
								else:
									lex_name = token

								tokens_set.add(token)
								tokens_pos_set.add(token_pos)
								lemma_pos_set.add(lemma_pos)								
								pos_set.add(pos)
	
								cand_pos[lemma_pos].add(pos)

								token_sentence.append(token)
								token_pos_sentence.append(token_pos)
								lemma_pos_sentence.append(lemma_pos)
								fine_grain_sentence.append(bn_syns)
								coarse_domain_sentence.append(domain)
								coarse_lex_sentence.append(lex_name)
								pos_sentence.append(pos)
								instance_sentence.append(1)

							else:
								print("------------------------------------------------------ ID Bn IN MAP ------------------------------------------------------")
		
						else:
							print("------------------------------------------------------ ID MISSING ------------------------------------------------------")
		corp_token.append(token_sentence)
		corp_token_pos.append(token_pos_sentence)
		corp_lemma_pos.append(lemma_pos_sentence)
		corp_fine_grain.append(fine_grain_sentence)
		corp_coarse_domain.append(coarse_domain_sentence)
		corp_coarse_lex.append(coarse_lex_sentence)
		corp_pos.append(pos_sentence)
		corp_instance.append(instance_sentence)


	# ------------------------------------------------------ Training dataset ------------------------------------------------------ 
	
	# print("Writing output file ...")
	# #Saving input and output files - strings
	# save_pickle(corp_token, "pickles/corpa/corp_token.pickle")
	# save_pickle(corp_token_pos, "pickles/corpa/corp_token_pos.pickle")
	# save_pickle(corp_lemma_pos, "pickles/corpa/corp_lemma_pos.pickle")
	# save_pickle(corp_fine_grain, "pickles/corpa/corp_fine_grain.pickle")
	# save_pickle(corp_coarse_domain, "pickles/corpa/corp_coarse_domain.pickle")
	# save_pickle(corp_coarse_lex, "pickles/corpa/corp_coarse_lex.pickle")
	# save_pickle(corp_pos, "pickles/corpa/corp_pos.pickle")
	# save_pickle(corp_instance, "pickles/corpa/corp_instanceùà.pickle")

	# #Saving sets - strings
	# save_pickle(tokens_set, "pickles/sets/tokens_set.pickle")
	# save_pickle(tokens_pos_set, "pickles/sets/tokens_pos_set.pickle")
	# save_pickle(lemma_pos_set, "pickles/sets/lemma_pos_set.pickle")
	# save_pickle(fine_senses_set, "pickles/sets/fine_senses_set.pickle")
	# save_pickle(coarse_domain_set, "pickles/sets/coarse_domain_set.pickle")
	# save_pickle(coarse_lex_set, "pickles/sets/coarse_lex_set.pickle")
	# save_pickle(pos_set, "pickles/sets/pos_set.pickle")

	# #Saving candidate labels dicts - strings
	# save_pickle(cand_fine_senses, "pickles/cand_dicts/cand_fine_senses.pickle")
	# save_pickle(cand_coarse_domain, "pickles/cand_dicts/cand_coarse_domain.pickle")
	# save_pickle(cand_coarse_lex, "pickles/cand_dicts/cand_coarse_lex.pickle")
	# save_pickle(cand_pos, "pickles/cand_dicts/cand_pos.pickle")


	# ------------------------------------------------------ dev/test dataset ------------------------------------------------------ 
	
	# print("Writing output file ...")
	# #Saving input and output files - strings
	# save_pickle(corp_token, "pickles/corpa/dev_corp_token.pickle")
	# save_pickle(corp_twoken_pos, "pickles/corpa/dev_corp_token_pos.pickle")
	# save_pickle(corp_lemma_pos, "pickles/corpa/dev_corp_lemma_pos.pickle")
	# save_pickle(corp_fine_grain, "pickles/corpa/dev_corp_fine_grain.pickle")
	# save_pickle(corp_coarse_domain, "pickles/corpa/dev_corp_coarse_domain.pickle")
	# save_pickle(corp_coarse_lex, "pickles/corpa/dev_corp_coarse_lex.pickle")
	# save_pickle(corp_pos, "pickles/corpa/dev_corp_pos.pickle")
	# save_pickle(corp_instance, "pickles/corpa/dev_instance.pickle")


if __name__ == '__main__':
	args = parse_args()
	semcor_parse(args.input_path, args.output_path, args.resources_path)
	# resources_path = "../resources"
	# input_path = "semcor.data.xml"
	# output_path = "../resources/output.txt"

	start = time.time()
	semcor_parse(input_path, output_path, resources_path)
	end = time.time()
	print("time:", end - start)
