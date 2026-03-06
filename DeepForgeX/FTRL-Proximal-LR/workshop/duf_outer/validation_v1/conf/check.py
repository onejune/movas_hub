import os,sys

fea_dic = {}
for line in sys.stdin:
    arr = line.strip().split('#')
    arr.sort()
    t = '#'.join(arr)
    if t in fea_dic:
        print(line.strip(), fea_dic[t])
        continue
    else:
        fea_dic[t] = line
    #print(t)
