
import pandas as pd

def ranking_recall(score_list):

    dataframe = pd.DataFrame()

    for score in score_list:
        dataframe = pd.concat([dataframe, pd.DataFrame(score)])
    
    dataframe['index'] = dataframe.index
    dataframe = dataframe[["model","index","0","1","accuracy","macro avg","weighted avg"]]

    dataframe['0'] = dataframe['0'].round(2)
    dataframe['1'] = dataframe['1'].round(2)
    dataframe['accuracy'] = dataframe['accuracy'].round(2)
    dataframe['macro avg'] = dataframe['macro avg'].round(2)
    dataframe['weighted avg'] = dataframe['weighted avg'].round(2)

    index = 'recall'

    dataframe_2 = dataframe.query(f"index == '{index}'")
    dataframe_2.head()

    df = dataframe_2.sort_values('1', ascending=False)
    return df