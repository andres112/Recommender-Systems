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

# Compute the prediction based on the 
def get_prediction(top):
    prediction = pd.DataFrame(columns=['movie','rating'])
    for movie in top.index.values:
        if(similarity.loc[0,movie+'.1'] > 0):
            # Select the films with the highest rating
            duplicateDFRow = similarity[[movie,movie+'.1']][similarity[movie+'.1'] == similarity.loc[0,movie+'.1']]
            duplicateDFRow['movie'] = duplicateDFRow[movie]
            duplicateDFRow['rating'] = duplicateDFRow[movie+'.1'] * top.loc[movie] # Predicted Ratings Calculation
            
            prediction = pd.concat([prediction, duplicateDFRow[['movie', 'rating']]], ignore_index=True)
    return prediction

# TASK related with Table 1
# AgrS: LEAST MISERY for thw whole Table 1
print("*****************************\nTask Number 1\n")
Group_recomendation={}
for group in Groups:
    top = ratings[Groups[group]].min(axis=1).sort_values(ascending=False)[:10]
    prediction = get_prediction(top)
    Group_recomendation[group] = prediction.sort_values('rating', ascending=False).drop_duplicates(subset ="movie")[:5]
    # print(f'\nGroup {group}:\n',Group_recomendation[group])

# TASK related with Table 2

print("\n*****************************\nTask Number 2\n")
# Only use the U5 and U15
Groups = {
    5: ['U31', 'U32', 'U33', 'U34', 'U40', 'U42', 'U18', 'U21', 'U48'],
    15: ['U39', 'U45', 'U47']
}
# AgrS: AVERAGE STRATEGY
Group_recomendation={}
print("\n* Average STRATEGY\n")
for group in Groups:
    # Get mean rating between users of the group
    top = ratings[Groups[group]].mean(axis=1).sort_values(ascending=False)[:10]
    prediction = get_prediction(top)
    Group_recomendation[group] = prediction.sort_values('rating', ascending=False).drop_duplicates(subset ="movie")[:5]
    print(f'\nGroup {group}:\n',Group_recomendation[group])

# AgrS: Most Pleasure STRATEGY
Group_recomendation={}
print("\n* Most Pleasure STRATEGY\n")
for group in Groups:
    # Get the max rating between users of the group
    top = ratings[Groups[group]].max(axis=1).sort_values(ascending=False)[:10]
    prediction = get_prediction(top)
    Group_recomendation[group] = prediction.sort_values('rating', ascending=False).drop_duplicates(subset ="movie")[:5]
    print(f'\nGroup {group}:\n',Group_recomendation[group])

# AgrS: Average without misery STRATEGY
Group_recomendation={}
print("\n* Average without misery STRATEGY\n")
# Define the threshold to define the misery value
threshold = 2
for group in Groups:
    # First get the users without misery value
    wo_misery = (ratings[Groups[group]] < threshold).sum()
    index = wo_misery[wo_misery == 0].index.values
    # Just computing the group predictions if there are users
    if(len(index) > 0):
        # Get mean rating between users of the group
        top = ratings[index].mean(axis=1).sort_values(ascending=False)[:10]
        prediction = get_prediction(top)
        Group_recomendation[group] = prediction.sort_values('rating', ascending=False).drop_duplicates(subset ="movie")[:5]
        print(f'\nGroup {group}:\n',Group_recomendation[group])
    else:
        print(f'\nGroup {group}:\nAll user of the group has at leas one movie rated under the threshold.')