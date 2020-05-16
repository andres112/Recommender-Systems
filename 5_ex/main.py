import pandas as pd

ratings = pd.read_csv('ratings.csv', sep=';', header=0, index_col='Movie')
similarity = pd.read_csv('similarity.csv', sep=';')

Groups = {
    1: ['U10', 'U26', 'U30', 'U12', 'U11', 'U16', 'U37', 'U29', 'U36'],
    2: ['U2', 'U3', 'U28', 'U44', 'U41'],
    3: ['U11', 'U15', 'U29'],
    4: ['U22', 'U50', 'U48'],
    5: ['U31', 'U32', 'U33', 'U34', 'U40', 'U42', 'U18', 'U21', 'U48'],
    6: ['U7', 'U8', 'U24', 'U25', 'U44'],
    7: ['U1', 'U6', 'U29'],
    8: ['U14', 'U27', 'U48'],
    9: ['U23', 'U10', 'U16', 'U36', 'U48'],
    10: ['U4', 'U20', 'U28', 'U47', 'U46', 'U45', 'U39', 'U49', 'U19'],
    11: ['U17', 'U2', 'U43', 'U5', 'U47'],
    12: ['U20', 'U28', 'U48', 'U35', 'U4'],
    13: ['U1', 'U51', 'U52', 'U53', 'U54', 'U55', 'U56', 'U57', 'U58'],
    14: ['U4', 'U19', 'U45', 'U46', 'U49'],
    15: ['U39', 'U45', 'U47']
}

def get_prediction():
    prediction = pd.DataFrame(columns=['movie','rating'])
    for movie in top.index.values:
        if(similarity.loc[0,movie+'.1'] > 0):
            duplicateDFRow = similarity[[movie,movie+'.1']][similarity[movie+'.1'] == similarity.loc[0,movie+'.1']]
            duplicateDFRow['movie'] = duplicateDFRow[movie]
            duplicateDFRow['rating'] = duplicateDFRow[movie+'.1'] * top.loc[movie]
            
            prediction = pd.concat([prediction, duplicateDFRow[['movie', 'rating']]])
    return prediction

# AgrS: LEAST MISERY
Group_recomendation={}
for group in Groups:
    top = ratings[Groups[group]].min(axis=1).sort_values(ascending=False)[:10]
    prediction = get_prediction()
    Group_recomendation[group] = prediction.sort_values('rating', ascending=False).drop_duplicates(subset ="movie")[:5]
    print(f'\nGroup {group}:\n',Group_recomendation[group])
