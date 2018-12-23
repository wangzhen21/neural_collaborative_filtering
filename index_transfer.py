#coding=utf-8
#获得index转换，根据用户ID和movieID 确定
import cPickle
from tqdm import tqdm
usermovie_newid = {}
oldmovie_newid = {}
f = open("Data/ratings.dat","rb")
index = 0
for line in tqdm(f):
    line_split = line.strip().split("::")
    if int(line_split[1]) not in oldmovie_newid.keys():
        oldmovie_newid[int(line_split[1])] = index
        index += 1

newid_oldid = {}
for key,val in oldmovie_newid.items():
    newid_oldid[val] = key

cPickle.dump(oldmovie_newid,open("Data/oldmovie_newid.p", "wb"))
cPickle.dump(newid_oldid,open("Data/newid_oldid.p", "wb"))