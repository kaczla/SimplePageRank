#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
PageRank (prosty) na osobach znajdujących się w polskiej Wikipedii.

created in 2017
@kaczla
"""

from __future__ import print_function
import re
import sys
import getopt
import time
import pickle
import lxml.etree as ET
import numpy as np
import cudamat as cm
# # example: http://www.cs.toronto.edu/%7Evmnih/docs/cudamat_tr.pdf
#import pyviennacl as pyv
# # example: https://www.karlrupp.net/2014/02/pyviennacl-gpu-accelerated-linear-algebra-for-python/



# BEGIN: USER VARIABLES
# If element MAX_MATRIX_ELEM is not equal 0, then is used MAX_MATRIX_ELEM_PERCENT, otherwise is used MAX_MATRIX_ELEM
#   MAX_MATRIX_ELEM         - Maximum matrix element in square matrix: MAX_MATRIX_ELEM x MAX_MATRIX_ELEM elements
#   MAX_MATRIX_ELEM_PERCENT - Maximum matrix element in percent (%), where maximum elements is numbers of people in Wiki: PEOPLE_IN_WIKI * MAX_MATRIX_ELEM_PERCENT (elements in square matrix is equal: PEOPLE_IN_WIKI * MAX_MATRIX_ELEM_PERCENT x PEOPLE_IN_WIKI * MAX_MATRIX_ELEM_PERCENT)
MAX_MATRIX_ELEM = 0
MAX_MATRIX_ELEM_PERCENT = 0.25
# MAX_MATRIX_POWER          - How many times power matrix 
MAX_MATRIX_POWER = 3
# END: USER VARIABLES

REGEX_BIBL_NOTES = re.compile(r'^Noty biograficzne -')
REGEX_URL = re.compile(r'\[{2}.*?\]{2}')
REGEX_AUTHOR_LINE = re.compile(r'^([:*|]|(Plik|File):)')
REGEX_IGNORE_AFTER_LINE = re.compile(r'^== Zobacz też ==$|^\[{2}Kategoria:')
REGEX_GET_AUTHOR_FORM_URL = re.compile(r'^\[{2}([^#|]+).*\]{2}$')
REGEX_REDIRECT = re.compile(r'^#(PATRZ|REDIRECT) ')

WIKI_FILE_NAME = 'plwiki-pages-articles.xml'
WIKI_NAMESPACE = '{http://www.mediawiki.org/xml/export-0.10/}'
WIKI_TAG_PAGE = WIKI_NAMESPACE + 'page'
WIKI_TAG_TITLE = WIKI_NAMESPACE + 'title'
WIKI_TAG_REVISION = WIKI_NAMESPACE + 'revision'
WIKI_TAG_TEXT = WIKI_NAMESPACE + 'text'

WIKI_AUTHOR_ID_LIST = dict()

PAGERANK_PROB_OTHER = 1/10
PAGERANK_PROB_SUCC = 9/10

MATRIX = None
MATRIX_MAPPER = dict()

MAX_RESULT = 10
ARGS_LIST = set()

class Author(object):
    """
    Klasa opisująca pojedyńczą osobę:
       name - nazwa osoby
       found - czy posiada stronę (poświęconą swojej bibliografi) na wiki:
                  False: oznacza, że nie posiada strony, istnieje tylko jako odnośnik;
                  True:  oznacza, że posiada strone;
       back_reference - ilość referencji odnoszących się do danej osoby (bierzący obiekt - osoba)
       back_reference_list - kto odnosi się do danej osoby
       forward_reference - ilość referencji, które odnoszą do innej osoby
       forward_reference_list - do kogo się odnosi
       aliases - inna nazwa osoby
    *****
    Uwaga - poniżej znajdują się dane, które nie zostały sklasyfikowane jako osoba, a istnieją jako strony w Wiki:
    *****
       back_reference_other - ilość referencji, które odnoszą się do bierzącej osoby i nie są osobami (inne podstrony)
       back_reference_other_list - co (nie kto!) odnosi się do danej osoby
       aliases_other - inne nazwy osoby
    """

    def __init__(self, _title):
        self.title = _title
        self.found = False
        self.back_reference = 0
        self.back_reference_list = set()
        self.forward_reference = 0
        self.forward_reference_list = set()
        self.aliases = set()
        self.back_reference_other = 0
        self.back_reference_other_list = set()
        self.aliases_other = set()

    def save(self, xml_doc=None):
        """
        Tworzy stronę XML.
        """
        if xml_doc is not None:
            page = ET.SubElement(xml_doc, 'page')
            #title
            ET.SubElement(page, 'title').text = self.title
            #found
            ET.SubElement(page, 'found').text = str(self.found)
            #back_reference
            ET.SubElement(page, 'back_reference').text = str(self.back_reference)
            xml_back_ref = ET.SubElement(page, 'back_reference_list')
            for ibackref in self.back_reference_list:
                ET.SubElement(xml_back_ref, 'back_reference_item').text = ibackref
            #forward_reference
            ET.SubElement(page, 'forward_reference').text = str(self.forward_reference)
            xml_forward_ref = ET.SubElement(page, 'forward_reference_list')
            for iforwardref in self.forward_reference_list:
                ET.SubElement(xml_forward_ref, 'forward_reference_item').text = iforwardref
            #aliases
            xml_alias = ET.SubElement(page, 'aliases')
            for ialias in self.aliases:
                ET.SubElement(xml_alias, 'alias').text = ialias
            #back_reference_other
            ET.SubElement(page, 'back_reference_other').text = str(self.back_reference_other)
            xml_back_ref_other = ET.SubElement(page, 'back_reference_other_list')
            for ibackref in self.back_reference_other_list:
                ET.SubElement(xml_back_ref_other, 'back_reference_other_item').text = ibackref
            #aliases_other
            xml_alias_other = ET.SubElement(page, 'aliases_other')
            for ialias in self.aliases_other:
                ET.SubElement(xml_alias_other, 'alias_other').text = ialias

def stage_1(save=False):
    """
    Pobranie listy osób, które mają note bibliograficzną.
       save - zapisanie postępu pracy STAGE_1
    """
    start_time = time.time()
    print("STAGE 1: BEGIN")
    save_file_name = 'stage_1.xml'
    print("Loading: %s" %(WIKI_FILE_NAME))
    if save:
        xml_save_stage_1 = ET.Element('document')
    #printing ignore url;
    #use: ./run.py > out_tmp.txt
    #then use: grep.sh with with function: author_outed
    print_ignored_url = False
    for _, element in ET.iterparse(WIKI_FILE_NAME, events=("end",), tag=WIKI_TAG_PAGE):
        # print(str(ET.tostring(element), 'utf-8'))
        found = False
        tmp_title = ''
        for child in element:
            if child.tag == WIKI_TAG_TITLE:
                if REGEX_BIBL_NOTES.match(child.text):
                    found = True
                    tmp_title = child.text
            elif found and child.tag == WIKI_TAG_REVISION:
                tmp_text = child.find(WIKI_TAG_TEXT).text
                if save:
                    xml_save_page = ET.SubElement(xml_save_stage_1, 'page')
                    ET.SubElement(xml_save_page, 'title').text = tmp_title
                    xml_save_page_text = ET.SubElement(xml_save_page, 'text')
                    xml_save_page_author = ET.SubElement(xml_save_page, 'authors')
                for line in tmp_text.split('\n'):
                    line = line.strip()
                    if line:
                        if REGEX_IGNORE_AFTER_LINE.match(line):
                            break
                        if save:
                            ET.SubElement(xml_save_page_text, 'p').text = line
                        if REGEX_AUTHOR_LINE.match(line):
                            authors = REGEX_URL.findall(line)
                            if authors:
                                author = REGEX_GET_AUTHOR_FORM_URL.match(authors[0]).group(1).upper()
                                if author not in WIKI_AUTHOR_ID_LIST:
                                    WIKI_AUTHOR_ID_LIST[author] = Author(author)
                                if save:
                                    ET.SubElement(xml_save_page_author, 'author').text = author 
                                #printing ignore url;
                                if print_ignored_url:
                                    if len(authors) > 1:
                                        for iauthor in authors:
                                            print('[2] Found: "%s"'%(iauthor))
                        else:
                            #printing ignore url;
                            if print_ignored_url:
                                authors = REGEX_URL.findall(line)
                                for iauthor in authors:
                                    print('[1] Found: "%s"'%(iauthor))
        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]
    print("Loaded:  %s" %(WIKI_FILE_NAME))
    if save:
        print("Saving: %s" %(save_file_name))
        tree = ET.ElementTree(xml_save_stage_1)
        tree.write(save_file_name, encoding='utf-8', xml_declaration=True, pretty_print=True)
        print("Saved:  %s" %(save_file_name))
    print("STAGE 1: END")
    print('TIME END: %s s' %(time.time() - start_time))

def stage_2(save=False, save_pickle=False, search_all=False):
    """
    Zliczenie ilości wystąpień - odnośników w innych dokumentach (domyślnie tylko pomiedzy osobami)
       save - zapisanie postępu pracy STAGE_2
       search_all - zliczanie we wszystkich dokumentów odnośników do osób (osobny licznik)
    """
    start_time = time.time()
    print("STAGE 2: BEGIN")
    if len(WIKI_AUTHOR_ID_LIST) <= 0:
        print('[ERR] Data not loaded!')
        return
    save_file_name = 'stage_2.xml'
    print("Loading: %s" %(WIKI_FILE_NAME))
    #printing author not found in wiki;
    #use: ./run.py > out_tmp.txt
    #then use: grep.sh with with function: author_notfound
    print_notfound_author = False
    for _, element in ET.iterparse(WIKI_FILE_NAME, events=("end",), tag=WIKI_TAG_PAGE):
        # print(str(ET.tostring(element), 'utf-8'))
        found = False
        tmp_title = ''
        for child in element:
            if child.tag == WIKI_TAG_TITLE:
                found = True
                tmp_title = child.text.upper()
                if tmp_title in WIKI_AUTHOR_ID_LIST:
                    WIKI_AUTHOR_ID_LIST[tmp_title].found = True
            elif found and child.tag == WIKI_TAG_REVISION:
                tmp_text = child.find(WIKI_TAG_TEXT).text
                if tmp_title and tmp_text:
                    author_page = False
                    if tmp_title in WIKI_AUTHOR_ID_LIST:
                        author_page = True
                    if REGEX_REDIRECT.match(tmp_text):
                        alias = REGEX_GET_AUTHOR_FORM_URL.match(REGEX_URL.search(tmp_text).group(0)).group(1).upper()
                        if alias in WIKI_AUTHOR_ID_LIST:
                            WIKI_AUTHOR_ID_LIST[alias].aliases.add(tmp_title)
                    if author_page or search_all:
                        for iauthor in set(REGEX_URL.findall(tmp_text)):
                            _match = REGEX_GET_AUTHOR_FORM_URL.match(iauthor)
                            if _match:
                                iauthor = _match.group(1).upper()
                                if iauthor in WIKI_AUTHOR_ID_LIST:
                                    if author_page:
                                        WIKI_AUTHOR_ID_LIST[iauthor].back_reference += 1
                                        WIKI_AUTHOR_ID_LIST[iauthor].back_reference_list.add(tmp_title)
                                        WIKI_AUTHOR_ID_LIST[tmp_title].forward_reference += 1
                                        WIKI_AUTHOR_ID_LIST[tmp_title].forward_reference_list.add(iauthor)
                                    elif search_all:
                                        WIKI_AUTHOR_ID_LIST[iauthor].back_reference_other += 1
                                        WIKI_AUTHOR_ID_LIST[iauthor].back_reference_other_list.add(tmp_title)
        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]
    print("Loaded:  %s" %(WIKI_FILE_NAME))
    if print_notfound_author:
        for key, value in WIKI_AUTHOR_ID_LIST.items():
            if not value.found:
                print('[3] Found: "%s"' %(key))
    if save:
        print("Saving: %s" %(save_file_name))
        xml_save_stage_2 = ET.Element('document')
        for key in sorted(WIKI_AUTHOR_ID_LIST):
            WIKI_AUTHOR_ID_LIST[key].save(xml_save_stage_2)
        tree = ET.ElementTree(xml_save_stage_2)
        tree.write(save_file_name, encoding='utf-8', xml_declaration=True, pretty_print=True)
        print("Saved:  %s" %(save_file_name))
        txt_file = open("out_ranking_back_reference.txt", "w")
        for key in sorted(WIKI_AUTHOR_ID_LIST, key=lambda name: WIKI_AUTHOR_ID_LIST[name].back_reference, reverse=True):
            txt_file.write("%s\t%d\n" %(key, WIKI_AUTHOR_ID_LIST[key].back_reference))
        txt_file.close()
        txt_file = open("out_ranking_forward_reference.txt", "w")
        for key in sorted(WIKI_AUTHOR_ID_LIST, key=lambda name: WIKI_AUTHOR_ID_LIST[name].forward_reference, reverse=True):
            txt_file.write("%s\t%d\n" %(key, WIKI_AUTHOR_ID_LIST[key].forward_reference))
        txt_file.close()
        txt_file = open("out_ranking_back_reference_other.txt", "w")
        for key in sorted(WIKI_AUTHOR_ID_LIST, key=lambda name: WIKI_AUTHOR_ID_LIST[name].back_reference_other, reverse=True):
            txt_file.write("%s\t%d\n" %(key, WIKI_AUTHOR_ID_LIST[key].back_reference_other))
        txt_file.close()
        txt_file = open("out_ranking_back_reference_sum.txt", "w")
        for key in sorted(WIKI_AUTHOR_ID_LIST, key=lambda name: (WIKI_AUTHOR_ID_LIST[name].back_reference + WIKI_AUTHOR_ID_LIST[name].back_reference_other), reverse=True):
            txt_file.write("%s\t%d\n" %(key, WIKI_AUTHOR_ID_LIST[key].back_reference + WIKI_AUTHOR_ID_LIST[key].back_reference_other))
        txt_file.close()
    if save_pickle:
        pickle.dump(WIKI_AUTHOR_ID_LIST, open('out_pickle_wiki_author.bin', 'wb'))
    print("STAGE 2: END")
    print('TIME END: %s s' %(time.time() - start_time))

def stage_3(save=False, save_pickle=False):
    """
    Przeliczanie PageRank - obliczanie macierzy przejscia
    """
    print("STAGE 3: BEGIN")
    if len(WIKI_AUTHOR_ID_LIST) <= 0:
        print('[ERR] Data not loaded!')
        return
    start_time = time.time()
    print('--row    : %d' %(int(MAX_MATRIX_ELEM)))
    print('--percent: %d (%f%%)' %(int(MAX_MATRIX_ELEM_PERCENT * len(WIKI_AUTHOR_ID_LIST)), MAX_MATRIX_ELEM_PERCENT*100))
    MATRIX_SIZE = int(MAX_MATRIX_ELEM_PERCENT * len(WIKI_AUTHOR_ID_LIST))
    if MATRIX_SIZE <= 0:
        MATRIX_SIZE = int(1)
    if MAX_MATRIX_ELEM > MATRIX_SIZE:
        MATRIX_SIZE = int(MAX_MATRIX_ELEM)
    if MATRIX_SIZE > len(WIKI_AUTHOR_ID_LIST):
        MATRIX_SIZE = int(len(WIKI_AUTHOR_ID_LIST))
    counter = 0
    for key in sorted(WIKI_AUTHOR_ID_LIST, key=lambda name: WIKI_AUTHOR_ID_LIST[name].back_reference, reverse=True):
        MATRIX_MAPPER[counter] = (key, np.float64(0.0), 0)
        counter += 1
    print("MATRIX SIZE: %d" %(MATRIX_SIZE))
    print("INIT MATRIX: BEGIN")
    global MATRIX
    MATRIX = np.zeros(shape=(MATRIX_SIZE, MATRIX_SIZE))
    print("INIT MATRIX: END")
    print("FILL MATRIX: BEGIN")
    fill_time = time.time()
    MATRIX_SHAPE = MATRIX.shape
    # 1 / n ; `n` - matrix dimension
    MATRIX_GO_RANDOM = 1/len(MATRIX_MAPPER)
    # 1/10 * 1/n ; `n` - matrix dimension
    MATRIX_GO_RANDOM_PROB = PAGERANK_PROB_OTHER*MATRIX_GO_RANDOM
    for i in range(MATRIX_SHAPE[0]):
        # count how many reference is not omitted by matrix size
        for_ref_list = WIKI_AUTHOR_ID_LIST[MATRIX_MAPPER[i][0]].forward_reference_list
        for_ref = 0
        for ikey, ivalue in MATRIX_MAPPER.items():
            if ivalue[0] in for_ref_list:
                for_ref += 1
        MATRIX_MAPPER[i] = (MATRIX_MAPPER[i][0], np.float64(0.0), for_ref)
        # if `i` have forward reference
        if for_ref > 0:
            for j in range(MATRIX_SHAPE[1]):
                # if exist reference from `i` to `j`
                if MATRIX_MAPPER[j][0] in for_ref_list:
                    # 9/10 * 1/x + 1/10 * 1/n ; n - matrix dimension ; x - numer of forward reference
                    MATRIX[i][j] = PAGERANK_PROB_SUCC*(1/for_ref) + MATRIX_GO_RANDOM_PROB
                # if not exist reference from `i` to `j`
                else:
                    # 1/10 * 1/n ; `n` - matrix dimension
                    MATRIX[i][j] = MATRIX_GO_RANDOM_PROB
        # if `i` not have forward reference
        else:
            for j in range(MATRIX_SHAPE[1]):
                # 1/n ; `n` - matrix dimension
                MATRIX[i][j] = MATRIX_GO_RANDOM
    fill_time = time.time() - fill_time
    print("FILL MATRIX: END   (TIME = %s s)" %(fill_time))
    if 'gpu' in ARGS_LIST:
        # cm
        MATRIX_GPU = cm.CUDAMatrix(MATRIX)
        # pyv
        #MATRIX_GPU = pyv.Matrix(MATRIX)
    else:
        MATRIX = np.matrix(MATRIX)
    print('--power  : %d' %MAX_MATRIX_POWER)
    print("POWER MATRIX: BEGIN")
    power_time = time.time()
    if 'gpu' in ARGS_LIST:
        for i in range(MAX_MATRIX_POWER-1):
            # pyv
            #MATRIX_GPU = MATRIX_GPU * MATRIX_GPU
            # cm
            #MATRIX_GPU = cm.dot(MATRIX_GPU, MATRIX_GPU)
            ## or
            cm.dot(MATRIX_GPU, MATRIX_GPU, MATRIX_GPU)
        # pyv
        #MATRIX = np.array(MATRIX_GPU.value)
        # cm
        MATRIX = MATRIX_GPU.asarray()
        del MATRIX_GPU
    else:
        MATRIX = MATRIX ** MAX_MATRIX_POWER
        MATRIX = np.array(MATRIX)
    power_time = time.time() - power_time
    print("POWER MATRIX: END  (TIME = %s s)" %(power_time))
    if save:
        txt_file_label = open("out_matrix_row_first_label.txt", "w")
        txt_file_data = open("out_matrix_row_first.txt", "w")
        txt_file_ref = open("out_matrix_row_first_ref.txt", "w")
        for i in range(MATRIX_SHAPE[0]):
            txt_file_label.write(MATRIX_MAPPER[i][0])
            txt_file_label.write("\n")
            txt_file_data.write('{0:.25f}'.format(MATRIX[0][i]))
            txt_file_data.write("\n")
            txt_file_ref.write(str(MATRIX_MAPPER[i][2]))
            txt_file_ref.write("\n")
            MATRIX_MAPPER[i] = (MATRIX_MAPPER[i][0], MATRIX[0][i], MATRIX_MAPPER[i][2])
        txt_file_label.close()
        txt_file_data.close()
        txt_file_ref.close()
    if save_pickle:
        pickle.dump(MATRIX_MAPPER, open('out_pickle_matrix_col.bin', 'wb'))
        pickle.dump(MATRIX, open('out_pickle_matrix.bin', 'wb'))
    del MATRIX
    print("STAGE 3: END")
    print('TIME END: %s s' %(time.time() - start_time))

def result():
    """
    Największy PageRank dla n-osób.
    """
    counter = 0
    print('\nRESULT:')
    for key in sorted(MATRIX_MAPPER, key=lambda name: MATRIX_MAPPER[name][1], reverse=True):
        counter += 1
        print(MATRIX_MAPPER[key][0], 'with probablity:', '{0:.25f}'.format(MATRIX_MAPPER[key][1]))
        if counter >= MAX_RESULT:
            break

def print_help():
    """
    Wyświetla pomoc.
    """
    print('Use: %s' %(sys.argv[0]))
    print('  -h --help          -> this help message\n')
    print('  -n                 -> how many people print as result (defualt: 10)')
    print('  -f --file [name]   -> Wikipedia file name (default: plwiki-pages-articles.xml)\n')
    print('  -p --pickle        -> use lib-pickle to save variables (default: disabled)')
    print('       + out_pickle_wiki_author.bin -> Wikipedia pages')
    print('       + out_pickle_matrix.bin      ->  PageRank matrix')
    print('       + out_pickle_matrix_col.bin  ->  matrix mapper: convert index to name\n')
    print('  -m --matrix        -> calculate matrix (ignore saved variable by pickle, if was used --pickle ; default: disabled)')
    print('  --row=[number]     -> max row in PageRank matrix (default: 0)')
    print('  --percent=[number] -> max percent row (where max row is equal: percer * numer of people in Wikipedia) in PageRank matrix (default: 0.25)')
    print('  if --row and --percent was added, then is chosen greater number of them')
    print('  --power=[number]   -> how many times power matrix (default: 3)\n')
    print('  -w --wiki          -> search in Wikipeda file (ignore saved variable by pickle, if was used --pickle ; default: disabled)\n')
    print('  --search           -> get extra information: reference page (not people page) to poeple')
    print('  -s --save          -> save all data to file (default: disabled)')
    print('       + stage_1.xml                          -> Wikipedia pages where was found people')
    print('       + stage_2.xml                          -> information about people')
    print('       + out_ranking_back_reference.txt       -> sorted list by back_reference, line format: `name [tab] value`')
    print('       + out_ranking_forward_reference.txt    -> sorted list by forward_reference, line format: `name [tab] value`')
    print('       + out_ranking_back_reference_other.txt -> sorted list by back_reference_other, line format: `name [tab] value`')
    print('       + out_ranking_back_reference_sum.txt   -> sorted list by (back_reference + back_reference_other), line format: `name [tab] value`')
    print('       + out_matrix_row_first_label.txt       -> label of first PageRank row, elements are separated by new line')
    print('       + out_matrix_row_first_ref.txt         -> how many reference have each element, elements are separated by new line ')
    print('       + out_matrix_row_first.txt             -> first PageRank row, elements are separated by new line\n')
    print('  --gpu            -> use GPU acceleration (using Nvidia CUDA)\n')

def read_args():
    """
    Wczytuje argumenty wejściowe.
    """
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hn:f:pmws", ["help", "file", "pickle", "matrix", "row=", "percent=", "power=", "wiki", "search", "save", "gpu"])
    except getopt.GetoptError as err:
        print_help()
        print(err)
        sys.exit(1)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
            sys.exit(0)
        elif opt in ("-n"):
            if int(arg) > 1:
                global MAX_RESULT
                MAX_RESULT = int(arg)
            else:
                print('Invalid -n value!')
                sys.exit(1)
        elif opt in ("-f", "--file"):
            global WIKI_FILE_NAME
            WIKI_FILE_NAME = arg
        elif opt in ("-p", "--pickle"):
            ARGS_LIST.add('pickle')
        elif opt in ("-m", "--matrix"):
            ARGS_LIST.add('matrix')
        elif opt == "--row":
            if int(arg) > 0:
                global MAX_MATRIX_ELEM
                MAX_MATRIX_ELEM = int(arg)
            else:
                print('Invalid --row value!')
                sys.exit(1)
        elif opt == "--percent":
            if float(arg) >= 0.0 and float(arg) <= 1.0:
                global MAX_MATRIX_ELEM_PERCENT
                MAX_MATRIX_ELEM_PERCENT = float(arg)
            else:
                print('Invalid --percent value!')
                sys.exit(1)
        elif opt == "--power":
            if int(arg) > 0:
                global MAX_MATRIX_POWER
                MAX_MATRIX_POWER = int(arg)
            else:
                print('Invalid --power value!')
                sys.exit(1)
        elif opt in ("-w", "--wiki"):
            ARGS_LIST.add('wiki')
        elif opt == '--search':
            ARGS_LIST.add('search')
        elif opt in ("-s", "--save"):
            ARGS_LIST.add('save')
        elif opt == '--gpu':
            ARGS_LIST.add('gpu')
        else:
            print_help()
            print('Unknown argument: \'%s\'' %opt)
            sys.exit(1)

if __name__ == "__main__":
    read_args()
    var_save_all = False
    var_save_pickle = False
    var_search_all = False
    if 'save' in ARGS_LIST:
        var_save_all = True
    if 'search' in ARGS_LIST:
        var_search_all = True
    if 'pickle' in ARGS_LIST:
        var_save_pickle = True
        if 'wiki' in ARGS_LIST:
            stage_1(save=var_save_all)
            stage_2(save=var_save_all, save_pickle=var_save_pickle, search_all=var_search_all)
        else:
            WIKI_AUTHOR_ID_LIST = pickle.load(open('out_pickle_wiki_author.bin', 'rb'))
            if not WIKI_AUTHOR_ID_LIST:
                stage_1(save=var_save_all)
                stage_2(save=var_save_all, save_pickle=var_save_pickle, search_all=var_search_all)
        if 'matrix' in ARGS_LIST:
            stage_3(save=var_save_all, save_pickle=var_save_pickle)
        else:
            MATRIX_MAPPER = pickle.load(open('out_pickle_matrix_col.bin', 'rb'))
            MATRIX = pickle.load(open('out_pickle_matrix.bin', 'rb'))
            if MATRIX.size <= 0 or not MATRIX_MAPPER:
                stage_3(save=var_save_all, save_pickle=var_save_pickle)
            else:
                del MATRIX
                del WIKI_AUTHOR_ID_LIST
    else:
        stage_1(save=var_save_all)
        stage_2(save=var_save_all, save_pickle=var_save_pickle, search_all=var_search_all)
        stage_3(save=var_save_all, save_pickle=var_save_pickle)
    result()
