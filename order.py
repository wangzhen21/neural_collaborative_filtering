#coding=utf-8
#将原本文件 按照时间 排序 ml-1m.test.rating.stance  ml-1m.train.rating.stance
import cPickle
from tqdm import tqdm
import utils.tools as tools

def outfile(file,list):
    with open(file,"aw+") as f:
        for item in list:
            f.write((item[0])+ "::" + item[1] + "::" + item[2] + "::" + item[3] + "\n")
def listout(foutlist_sum):
    train_foutlist = []
    test_foutlist = []
    foutlist_sum.sort(key=lambda x: x[3])
    if len(foutlist_sum) > 0:
        outfile("Data/ratings.sort.dat", foutlist_sum)
tools.deletefilesmovies(["Data/ratings.sort.dat"])
f = open("Data/ratings.dat","rb")

last_id = 1
foutlist_sum = []
for line in tqdm(f):
    line_split = line.strip().split("::")
    id = int(line_split[0])
    if id != last_id:
        last_id = id
        listout(foutlist_sum)
        foutlist_sum = []
    foutlist_sum.append(line_split)
listout(foutlist_sum)