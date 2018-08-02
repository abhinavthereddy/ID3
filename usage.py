from id3 import *

file = r'tennis.csv'
df = pd.read_csv('tennis.csv')


sample = ID3()
sample.create_tree(df)



file = r'tennis (test).csv'
df_test = pd.read_csv('tennis (test).csv')

df_predicted = sample.predict_outcome(df_test)
print(df_predicted)