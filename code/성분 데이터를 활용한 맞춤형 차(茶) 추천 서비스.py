#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')

name = sys.argv[1]
print(name + "python 실행 성공")

tea = pd.read_csv('C://Users/SM107/Desktop/new/teas.csv', encoding = 'EUC-KR')
tea.set_index('tea_id')


# In[ ]:


tea_df = tea[['tea_id','tea_name','efficacies', 'score_average', 'score_count']]
pd.set_option('max_colwidth', 100)


# In[ ]:


from ast import literal_eval

tea_df['efficacies'] = tea_df['efficacies'].apply(literal_eval)


# In[ ]:


tea_df['efficacies'] = tea_df['efficacies'].apply(lambda x : [ y['efficacy_id'] for y in x])


# In[ ]:


import cx_Oracle
user_list = []
def select(var):
    conn = cx_Oracle.connect("smhrd/aorwntkfkd1!@smhrdai.cunegl97a26d.us-east-2.rds.amazonaws.com/orcl")
    cursor = conn.cursor()
    
    sql = "select * from survey"
    cursor.execute(sql)

    for row in cursor :
        user_list.append(row)
        print(row)
    cursor.close()   
    conn.close()
select(())


# In[ ]:


ul = pd.DataFrame(user_list)

for i in range(len(ul[1])):
    a = ul[1]
    a[i] = a[i].replace("0","AAAX, AACO, AADF, AAAM, AAAP, AAAL")
    a[i] = a[i].replace("1","AAAF, AACG, AACI, AACN, AACS, AADB, AABI, AABR, AADJ, AABG, AADF, AAAM, AAAP, AAAL")
    a = ul[2]
    a[i] = a[i].replace("1","AADE")
    a = ul[3]
    a[i] = a[i].replace("1","AACT")
    a = ul[4]
    a[i] = a[i].replace("1","AABD, AADN")
    a = ul[5]
    a[i] = a[i].replace("1","AAAC, AAAN, AAAU, AADP, AACL, AADH, AAAW, AAAE")
    a = ul[6]
    a[i] = a[i].replace("1","AABE, AADT, AACW, AABO, AACU, AACR")
    a = ul[7]
    a[i] = a[i].replace("1","AAAH, AAAG, AAAR, AAAI, AAAV, AACZ, AADT, AACE, AAAO, AABK, AACB, AABZ, AACP, AADL, AAAZ")
    a = ul[8]
    a[i] = a[i].replace("1","AAAQ, AADO, AABK")
    a = ul[9]
    a[i] = a[i].replace("1","AACH, AABJ, AABP, AABA, AABB, AAAY, AABY")
    a = ul[10]
    a[i] = a[i].replace("1","AABT, AACA, AABW")
    a = ul[11]
    a[i] = a[i].replace("1","AABS, AACX, AABK")
    a = ul[12]
    a[i] = a[i].replace("1","AADA")
    a = ul[13]
    a[i] = a[i].replace("1","AAAJ")
    
    ef_list = []
    for x in range(len(ul)):
        a = list(ul.iloc[x,:])
        b = a.count('0')
        for y in range(b):
            a.remove('0')
        ef_list.append(a)
    
    EF = {}
    for i in range(len(ef_list)):
        EF[ef_list[i][0]]=ef_list[i][1:]
    
    H = pd.DataFrame((EF.keys(),EF.values()))
    H = H.T
    H.columns = ['tea_name', 'efficacies']


# In[ ]:


tea_df = pd.concat([tea_df, H],axis = 0, ignore_index=True)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

tea_df['efficacies_literal'] = tea_df['efficacies'].apply(lambda x : (' ').join(x))

count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
efficacy_mat = count_vect.fit_transform(tea_df['efficacies_literal'])
print(efficacy_mat.shape)


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity

efficacy_sim = cosine_similarity(efficacy_mat, efficacy_mat)
print(efficacy_sim.shape)
print(efficacy_sim[:1])


# In[ ]:


efficacy_sim_sorted_ind = efficacy_sim.argsort()[:, ::-1]
print(efficacy_sim_sorted_ind[:1])


# In[ ]:


def find_sim_tea1(df, sorted_ind, tea_name, top_n=10):
    
    title_tea = tea_df[tea_df['tea_name'] == tea_name]
    title_index = title_tea.index.values
    
    similar_indexes = sorted_ind[title_index, :(top_n)]
    similar_indexes = similar_indexes[similar_indexes != title_index]

    similar_indexes = similar_indexes.reshape(-1)
    
    return df.iloc[similar_indexes]


# In[ ]:


conn = cx_Oracle.connect("smhrd/aorwntkfkd1!@smhrdai.cunegl97a26d.us-east-2.rds.amazonaws.com/orcl")
str = [name]
    
similar_teas = find_sim_tea1(tea_df, efficacy_sim_sorted_ind, name ,10)
similar_teas = similar_teas.dropna(axis=0)
similar_teas['tea_id'] = pd.to_numeric(similar_teas['tea_id'])
similar_teas[['tea_id', 'tea_name', 'score_average']]
    
a = similar_teas['tea_id'][:5]

tea_1 = {'member_id' : str, 'tea_1' : a.values[0]}
survey_result_1 = pd.DataFrame((tea_1))
rows_1 = [tuple(x) for x in survey_result_1.to_records(index = False)]

tea_2 = {'member_id' : str, 'tea_2' : a.values[1]}
survey_result_2 = pd.DataFrame((tea_2))
rows_2 = [tuple(x) for x in survey_result_2.to_records(index = False)]

tea_3 = {'member_id' : str, 'tea_3' : a.values[2]}
survey_result_3 = pd.DataFrame((tea_3))
rows_3 = [tuple(x) for x in survey_result_3.to_records(index = False)]

tea_4 = {'member_id' : str, 'tea_4' : a.values[3]}
survey_result_4 = pd.DataFrame((tea_4))
rows_4 = [tuple(x) for x in survey_result_4.to_records(index = False)]

tea_5 = {'member_id' : str, 'tea_5' : a.values[4]}
survey_result_5 = pd.DataFrame((tea_5))
rows_5 = [tuple(x) for x in survey_result_5.to_records(index = False)]

cursor = conn.cursor()

print(rows_1)
sql = "insert into survey_result_1 values(:1, :2)"
cursor.executemany(sql, rows_1)  
print(cursor.rowcount, "record inserted.\n")

print(rows_2)
sql = "insert into survey_result_2 values(:1, :2)"
cursor.executemany(sql, rows_2)  
print(cursor.rowcount, "record inserted.\n")

print(rows_3)
sql = "insert into survey_result_3 values(:1, :2)"    
cursor.executemany(sql, rows_3)  
print(cursor.rowcount, "record inserted.\n")

print(rows_4)
sql = "insert into survey_result_4 values(:1, :2)"    
cursor.executemany(sql, rows_4)   
print(cursor.rowcount, "record inserted.\n")

print(rows_5)
sql = "insert into survey_result_5 values(:1, :2)"    
cursor.executemany(sql, rows_5)   
print(cursor.rowcount, "record inserted.\n")

cursor.close()             
conn.commit()
conn.close()


# In[ ]:


C = tea_df['score_average'].mean()
m = tea_df['score_count'].quantile(0.6)
print('C:',round(C,3), 'm:',round(m,3))


# In[ ]:


percentile = 0.6
m = tea_df['score_count'].quantile(percentile)
C = tea_df['score_average'].mean()

def weighted_vote_average(record):
    v = record['score_count']
    R = record['score_average']
    
    return ( (v/(v+m)) * R ) + ( (m/(m+v)) * C )   

tea_df['weighted_vote'] = tea_df.apply(weighted_vote_average, axis=1) 


# In[ ]:


def find_sim_tea2(df, sorted_ind, tea_name, top_n=10):
    title_tea = tea_df[tea_df['tea_name'] == tea_name]
    title_index = title_tea.index.values
    
    similar_indexes = sorted_ind[title_index, :(top_n*2)]
    similar_indexes = similar_indexes.reshape(-1)
    similar_indexes = similar_indexes[similar_indexes != title_index]
    
    return tea_df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]


# In[ ]:


conn = cx_Oracle.connect("smhrd/aorwntkfkd1!@smhrdai.cunegl97a26d.us-east-2.rds.amazonaws.com/orcl")
str = [name]
    
similar_teas = find_sim_tea2(tea_df, efficacy_sim_sorted_ind, name ,10)
similar_teas = similar_teas.dropna(axis=0)
similar_teas['tea_id'] = pd.to_numeric(similar_teas['tea_id'])
similar_teas[['tea_id', 'tea_name', 'score_average']]
    
a = similar_teas['tea_id'][:5]

tea_1 = {'member_id' : str, 'tea_1' : a.values[0]}
survey_result_score_1 = pd.DataFrame((tea_1))
rows_1 = [tuple(x) for x in survey_result_score_1.to_records(index = False)]

tea_2 = {'member_id' : str, 'tea_2' : a.values[1]}
survey_result_score_2 = pd.DataFrame((tea_2))
rows_2 = [tuple(x) for x in survey_result_score_2.to_records(index = False)]

tea_3 = {'member_id' : str, 'tea_3' : a.values[2]}
survey_result_score_3 = pd.DataFrame((tea_3))
rows_3 = [tuple(x) for x in survey_result_score_3.to_records(index = False)]

tea_4 = {'member_id' : str, 'tea_4' : a.values[3]}
survey_result_score_4 = pd.DataFrame((tea_4))
rows_4 = [tuple(x) for x in survey_result_score_4.to_records(index = False)]

tea_5 = {'member_id' : str, 'tea_5' : a.values[4 ]}
survey_result_score_5 = pd.DataFrame((tea_5))
rows_5 = [tuple(x) for x in survey_result_score_5.to_records(index = False)]

cursor = conn.cursor()

print(rows_1)
sql = "insert into survey_result_score_1 values(:1, :2)"
cursor.executemany(sql, rows_1)  
print(cursor.rowcount, "record inserted.\n")

print(rows_2)
sql = "insert into survey_result_score_2 values(:1, :2)"
cursor.executemany(sql, rows_2)   
print(cursor.rowcount, "record inserted.\n")

print(rows_3)
sql = "insert into survey_result_score_3 values(:1, :2)"    
cursor.executemany(sql, rows_3)  
print(cursor.rowcount, "record inserted.\n")

print(rows_4)
sql = "insert into survey_result_score_4 values(:1, :2)"    
cursor.executemany(sql, rows_4)  
print(cursor.rowcount, "record inserted.\n")

print(rows_5)
sql = "insert into survey_result_score_5 values(:1, :2)"    
cursor.executemany(sql, rows_5)   
print(cursor.rowcount, "record inserted.\n")

cursor.close()             
conn.commit()
conn.close()

