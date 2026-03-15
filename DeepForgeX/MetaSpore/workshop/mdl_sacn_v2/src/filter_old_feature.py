# -*- coding: utf-8 -*- 
'''
Created on 2017年8月21日

@author: thinkpad
'''
import sys
old_file,new_file=sys.argv[1:]
print(old_file, new_file)
his_data={}
with open(old_file,'r') as r:
    for i in r.readlines():
        info = i.rstrip().split('\002')
        key=info[0]
        imp=info[-1]
        click=info[-2]
        his_data[key]=(imp,click)
with open(new_file,'r') as r,open(new_file+'.tmp','w') as w:
    for i in r.readlines():
        info = i.rstrip().split('\002')
        key=info[0]
        imp=info[-1]
        click=info[-2]
        if key not in his_data:
            w.writelines(i)
            continue
        his_imp,his_click=his_data[key]
        if imp != his_imp or click != his_click:
            w.writelines(i)
        else:
            print('old_feature: ' ,'\t'.join(info))
            
        
if __name__ == '__main__':
    pass
