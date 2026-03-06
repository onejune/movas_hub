import os,sys

for line in sys.stdin:
    arr = line.strip().split('#')
    for ele in arr:
        print(ele)
