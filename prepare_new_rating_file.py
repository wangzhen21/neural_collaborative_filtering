#coding=utf-8
#将文件转化为新的ID 新文件为 ml-1m.test.rating.stance  ml-1m.train.rating.stance
import cPickle
from tqdm import tqdm
import utils.tools as tools 

def outfile(file,list):
    with open(file,"aw+") as f:
        for item in list:
            f.write(str(int(item[0]) - 1) + "\t" + str(item[1]) + "\t" + str(item[2]) + "\t" + str(item[3]) + "\t" + str(item[4])
                    + "\t" + str(item[5]) + "\n")
def listout(foutlist_sum):
    train_foutlist = []
    test_foutlist = []
    foutlist_sum.sort(key=lambda x: x[3],reverse=True)
    if len(foutlist_sum) > 0:
        test_foutlist.append(foutlist_sum[0])
        foutlist_sum.remove(foutlist_sum[0])
        outfile("Data/ml-1m.test.rating.stance", test_foutlist)
        outfile("Data/ml-1m.train.rating.stance", foutlist_sum)


oldmovie_newid = cPickle.load(open("Data/oldmovie_newid.p", "rb"))
newid_oldid = cPickle.load(open("Data/newid_oldid.p", "rb"))

f = open("Data/ratings_date_dire.dat","rb")
tools.deletefilesmovies(["Data/ml-1m.test.rating.stance","Data/ml-1m.train.rating.stance"])
tools.deletefilesmovies(["Data/rating_with_new_index.dat"])
last_id = 1
foutlist_sum = []
for line in tqdm(f):
    line_split = line.strip().split("::")
    line_split[1] = int(oldmovie_newid[int(line_split[1])])
    id = int(line_split[0])
    if id != last_id:
        last_id = id
        outfile("Data/rating_with_new_index.dat", foutlist_sum)
        listout(foutlist_sum)
        foutlist_sum = []
    foutlist_sum.append(line_split)
listout(foutlist_sum)