import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge

#Import Train Data
train = pd.read_csv('train.csv')
print(train.head())

#Clean Train Data
train['Personality'] = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})
train['Stage_fear'] = train['Stage_fear'].map({'No': 0, 'Yes': 1})
train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'No': 0, 'Yes': 1})

train = train.fillna({
    'Time_spent_Alone': train['Time_spent_Alone'].mean(),
    'Stage_fear': 0,
    'Social_event_attendance': train['Social_event_attendance'].mean(),
    'Going_outside': train['Going_outside'].mean(),
    'Drained_after_socializing': 0,
    'Friends_circle_size': train['Friends_circle_size'].mean(),
    'Post_frequency': train['Post_frequency'].mean(),
    'Personality': 0
})

# Split Training Data
X_train = train.drop(['id','Personality'], axis=1)
y_train = train['Personality']

#Scale Training Data
scaler = MinMaxScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# Create and train linear regression model / or Ridge on version 2
#model = LinearRegression()
model = Ridge(alpha=1.0)  # You can tune alpha (λ)
model.fit(train_scaled, y_train)



#Import Test Data
test = pd.read_csv('test.csv')
test_model = test.drop('id',axis=1)

#Clean Test Data
test_model['Stage_fear'] = test_model['Stage_fear'].map({'No': 0, 'Yes': 1})
test_model['Drained_after_socializing'] = test_model['Drained_after_socializing'].map({'No': 0, 'Yes': 1})

test_model = test_model.fillna({
    'Time_spent_Alone': test_model['Time_spent_Alone'].mean(),
    'Stage_fear': 0,
    'Social_event_attendance': test_model['Social_event_attendance'].mean(),
    'Going_outside': test_model['Going_outside'].mean(),
    'Drained_after_socializing': 0,
    'Friends_circle_size': test_model['Friends_circle_size'].mean(),
    'Post_frequency': test_model['Post_frequency'].mean()
})

#Scale test Data
test_model_scaled = scaler.transform(test_model)  # not fit_transform, just transform

# Make predictions
y_pred = model.predict(test_model_scaled)
y_pred = np.where(y_pred>=0.5, 'Extrovert', 'Introvert')

#Submission file
submission = test
submission['Personality']= y_pred
submission= submission.loc[:,['id','Personality']]
submission.to_csv('submission.csv',index=False)

print(model.coef_)