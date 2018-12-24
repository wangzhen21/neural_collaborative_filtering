#coding=utf-8
#将文件转化为新的ID 新文件为 ml-1m.test.rating.stance  ml-1m.train.rating.stance
import cPickle
from tqdm import tqdm
import cPickle
import utils.tools as tools
def outfile(file,stritem):
    with open(file,"aw+") as f:
            f.write(stritem + "\n")
newid_oldid = cPickle.load(open("Data/newid_oldid.p","rb"))
oldmovie_newid = cPickle.load(open("Data/oldmovie_newid.p","rb"))
movielid_dir_name_full_movie = cPickle.load(open("Data/movielid_dir_name_full_movie.p","rb"))
user_director_count_all = cPickle.load(open("Data/user_director_count_all.d","rb"))
tools.deletefilesmovies(["Data/ml-1m.test.negtive.flag.stance.newindex"])
test_id_index = {}
def get_saw_dire_num(user_id,movie_id_new):
    director_count = user_director_count_all[int(user_id)]
    movie_id_old = newid_oldid[movie_id_new]
    try:
        dir_index = movielid_dir_name_full_movie[movie_id_old][0]
    except:
        return "1"
    if dir_index in director_count.keys():
        return str(director_count[dir_index] + 1)
    else:
        return "1"
# with open("Data/ml-1m.test.rating.stance","rb") as f:
#     for line in f:
#         line_split = line.strip().split("\t")
#         test_id_index[int(line_split[0])] = line_split[1]
with open("Data/ml-1m.test.negtive.stance.newindex","rb") as f:
    for line in tqdm(f):
        later = ""
        line_split = line.strip().split(")")
        id_pair = line_split[0]
        user_id = (id_pair.split(","))[0][1:len((id_pair.split(","))[0])]
        negtive_movie_id_list = line_split[1].strip().split("\t")
        for item in negtive_movie_id_list:
            new_dire_count = get_saw_dire_num(user_id,int(item))
            later = later + "\t" + new_dire_count
        outfile("Data/ml-1m.test.negtive.flag.stance.newindex",id_pair + ")" + later)
