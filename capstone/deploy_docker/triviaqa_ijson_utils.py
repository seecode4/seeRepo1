#!/usr/bin/python
# # triviaqa_ijson_utils.py

# ijson utilities to get the qa entries from triviaqa json files into a list of dict items

import os
import json
import ijson
# ijson Version: 3.3.0


def get_json_filenames_linecnt(data_path):
    """
    Return .json filenames and linecnt in given data_path
    """
    files = os.listdir(data_path)
    fnumlines = {}
    # Iterate through the list and print the name of each file
    for file in files:
        # if os.path.isfile(os.path.join(directory_path, file)):
        if file.endswith(".json"):
            fname = data_path+'/'+file
            num_lines = sum(1 for line in open(fname))
            # fnumlines[fname] = num_lines
            fnumlines[file] = num_lines
    return (fnumlines)


def getqa_entries(kvitems, keys, numqa):
    """
    Create a list of numqa entries; each entry is a dict with keys and values
    if numqa < 1 get all entries.
    Only save the following as a dictionary entries
      Question
      NormalizedAliases, NormalizedValue and Value from Answer
    Return the qa_dict_list
    """
    one_qa_dict = {}
    qa_dict_list = []
    qacnt = 1
    kcnt = 0
    klen = len(keys)
    for k, v in kvitems:
        # cnt += 1
        if k in keys:
            kcnt += 1
            # Only save 'Answer' keys 'NormalizedAliases', 'NormalizedAliases'
            # and 'Question' key-values
            if k == 'Answer':
                one_qa_dict['NormalizedAliases'] = v['NormalizedAliases']
                one_qa_dict['NormalizedValue'] = v['NormalizedValue']
                one_qa_dict['Value'] = v['Value']
            elif k == 'Question':
                one_qa_dict[k] = v
            if kcnt == klen:
                qa_dict_list.append(one_qa_dict)
                one_qa_dict = {}
                kcnt = 0
                qacnt += 1
        if (numqa > 0) and (qacnt > numqa):
            break
    return qa_dict_list


def getqa_entries_from_file(fname, numqa):
    """
    Get a list of numqa entries from given file
    Each entry is a dict with keys and values
    if numqa < 1 get all entries
    """
    # File is organized as in keys_main; here Data contains the qa entries
    keys_main = ['Data', 'Domain', 'VerifiedEval', 'Version']
    qakeys = ['Answer', 'EntityPages', 'Question', 'QuestionId',
              'QuestionSource']
    with open(fname, 'r', encoding='utf-8') as fp:
        kv_items = ijson.kvitems(fp, 'Data.item')
        # print(type(kv_items))  # <class '_yajl2.kvitems'>
        getqa_res = getqa_entries(kv_items, qakeys, numqa)
        fp.close()
        return getqa_res


def print_sample_qa(qa_dict_list):
    """
    print some qa samples from given qa list
    """
    num_qa = len(qa_dict_list)
    for i in range(num_qa):
        if (i < 2) or (i % 1000 == 0) or (i > (num_qa-3)):
            qa_i = qa_dict_list[i]
            ans = 'TBD'
            if 'Value' in qa_i.keys():
                ans = qa_i['Value']
            print(f'{i:05}', qa_i['Question'], "...", ans)
    return num_qa


def create_datsets_dict_list(data_path):
    """
    Read json data files and for each create a dict with 'filename', 'linecnt',
    'qacnt' and 'qa_dict_list' which is a dict list of qa entries
    Collect these dict entries into a list
    """
    print("Reading files from ", data_path)
    # ijson parse events
    possible_events = ['start_map', 'end_map',
                       'start_array', 'end_array', 'map_key']
    fnumlines = get_json_filenames_linecnt(data_path)

    # keys in qa file
    keys_main = ['Data', 'Domain', 'VerifiedEval', 'Version']
    qakeys = ['Answer', 'EntityPages', 'Question', 'QuestionId',
              'QuestionSource']

    # if dbglevel > 0:
    #     print(fnumlines)

    # Read each file ; save qa into its qa_dict_list (a list of qa dict entries)
    datsets_dict_list = []
    for fname, lcnt in fnumlines.items():
        datsets_dict = {}
        qa_dict_list = getqa_entries_from_file(data_path+fname, lcnt)
        qacnt = len(qa_dict_list)
        datsets_dict = {'filename': fname, 'linecnt': lcnt,
                        'qacnt': qacnt, 'qa_dict_list': qa_dict_list}
        datsets_dict_list.append(datsets_dict)
    return datsets_dict_list


def dbg_print_qa_files_info(datsets_dict_list):
    numfiles = len(datsets_dict_list)
    print("Num json data files =", numfiles)
    for i in range(numfiles):
        finfo = datsets_dict_list[i]
        qa_dict_list = finfo['qa_dict_list']
        print('---------------------------------------')
        print(f"\nNum qa in {finfo['filename']}: {len(qa_dict_list)}",
              f"linecnt: {finfo['linecnt']}\n")
        qacnt = print_sample_qa(qa_dict_list)
    print('---------------------------------------')


def generate_data(data):
    """ use to yield dict entries one at a time """
    for item in data:
        yield item


def write_dict_list_to_json_file(file_name, dict_list, maxcnt):
    """
    Save to file as a list of dict; [{k:v, k1:v1..}, {k:v, k1:v1..}, ...]
    """
    cnt = 0
    with open(file_name, 'w') as fp:
        fp.write('[\n')
        first = True
        for item in generate_data(dict_list):
            if not first:
                fp.write(',\n')
            json.dump(item, fp)
            first = False
            cnt += 1
            if (cnt >= maxcnt):
                break
        fp.write('\n]')
    fp.close()
    return


def read_dict_list_from_json_file(file_name, maxcnt):
    """
    Read from file a list of dict; [{k:v, k1:v1..}, {k:v, k1:v1..}, ...]
    """
    cnt = 0
    dict_list = []
    with open(file_name, 'r') as fp:
        for item in ijson.items(fp, 'item'):
            dict_list.append(item)
            cnt += 1
            if (cnt >= maxcnt):
                break
        fp.close()
    return dict_list
