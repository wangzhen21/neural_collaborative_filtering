'''
author wangzhen
2018/12/13
'''
import os
import re


def deletefilesmovies(files):
    for item in files:
        if os.path.exists(item):
            os.remove(item)

def getfileitemdir(files,pre_index,la_index,pattern_pre,pattern_lat,pattern_content_pre,pattern_content_la,sperator,expel_first_line,num = True):
    '''

    :param files:
    :param pattern_pre:
        regular expression1
    :param pattern_lat:
        regular expression2
    :param sperator:
        , \t etc
    :param expel_first_line:
        ture or false, expel the first line
    :return:
        :return the directory key is the first instance index , value is second index
    '''
    dir = {}
    with open(files, "rb") as f:
        # transfer movieslens2imdb integer to integer
        for line in f:
            if expel_first_line:
                expel_first_line = False
                continue
            else:
                line_split = line.strip().split(sperator)
                pre_match = re.search(pattern_pre,line_split[pre_index])
                lat_match = re.search(pattern_lat,line_split[la_index])
                if pre_match != None:
                    if lat_match != None:
                        pre_content = re.search(pattern_content_pre,pre_match.group()).group()
                        lat_content = re.search(pattern_content_la,lat_match.group()).group()
                        m_lens = int(pre_content)
                        if num == True:
                            imdb_score = int(lat_content)
                        else:
                            imdb_score = lat_content
                        dir[m_lens] = imdb_score
    return dir

def judgefilelistexisting(files):
    '''
    judge file exsiting or not
    :param files:
    :return:
    '''
    for item in files:
        if not os.path.exists(item):
            return False
    return True

def judgefileexisting(files):
    '''
    judge file exsiting or not
    :param files:
    :return:
    '''
    if not os.path.exists(files):
            return False
    return True

def outfile_first(file,first_line):
    if os.path.exists(file):
        os.remove(file)
    with open(file, "wb+") as f:
            f.write(first_line + "\n")