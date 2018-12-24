#coding=utf-8
#将文件转化为新的ID 新文件为 ml-1m.test.rating.stance  ml-1m.train.rating.stance
import cPickle
from tqdm import tqdm
import utils.tools as tools
def outfile(file,stritem):
    with open(file,"aw+") as f:
            f.write(stritem + "\n")

tools.deletefilesmovies(["Data/ml-1m.test.negtive.stance.newindex"])
test_id_index = {}
with open("Data/ml-1m.test.rating.stance","rb") as f:
    for line in f:
        line_split = line.strip().split("\t")
        test_id_index[int(line_split[0])] = line_split[1]
with open("Data/ml-1m.test.negative","rb") as f:
    for line in f:
        line_split = line.strip().split(")")
        id_pair = line_split[0]
        id_pair = id_pair[1:len(line_split[0])]
        id_pair = (id_pair.split(","))[0]
        outfile("Data/ml-1m.test.negtive.stance.newindex", "(" + id_pair + "," + test_id_index[int(id_pair)] + ")" + line_split[1])