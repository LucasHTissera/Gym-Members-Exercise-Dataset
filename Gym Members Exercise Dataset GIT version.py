# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 19:26:36 2024

@author: Lucas
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import duckdb as ddb
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
#%%

data = pd.read_csv("gym_members_exercise_tracking.csv")

data.reset_index(inplace=True)

#%%

"""
Key Features:

Age: Age of the gym member.

Gender: Gender of the gym member (Male or Female).

Weight (kg): Member’s weight in kilograms.

Height (m): Member’s height in meters.

Max_BPM: Maximum heart rate (beats per minute) during workout sessions.

Avg_BPM: Average heart rate during workout sessions.

Resting_BPM: Heart rate at rest before workout.

Session_Duration (hours): Duration of each workout session in hours.

Calories_Burned: Total calories burned during each session.

WO_Type: Type of workout performed (e.g., Cardio, Strength, Yoga, HIIT).

Fat_Percentage: Body fat percentage of the member.

Water_Intake (liters): Daily water intake during workouts.

WO_Freq (days/week): Number of workout sessions per week.

Experience_Level: Level of experience, from beginner (1) to expert (3).

BMI: Body Mass Index, calculated from height and weight.
"""

con = ddb.connect()

con.register('data_', data)

con.execute("CREATE TABLE gym_members AS SELECT * FROM data_")

#%%
#Count amount of categorical attributes

diferent_workouts = con.sql("SELECT DISTINCT WO_Type FROM gym_members").to_df()

amnt_of_males = con.sql("""SELECT COUNT(index) AS Amount FROM gym_members
                        WHERE Gender = 'Male'""").to_df()

amnt_of_females = con.sql("""SELECT COUNT(index) AS Amount FROM gym_members
                        WHERE Gender = 'Female'""").to_df()
                        
amnt_of_yoga = con.sql("""SELECT COUNT(index) AS Amount FROM gym_members
                        WHERE WO_Type = 'Yoga'""").to_df()

amnt_of_HIIT = con.sql("""SELECT COUNT(index) AS Amount FROM gym_members
                        WHERE WO_Type = 'HIIT'""").to_df()

amnt_of_cardio = con.sql("""SELECT COUNT(index) AS Amount FROM gym_members
                        WHERE WO_Type = 'Cardio'""").to_df()

amnt_of_strength = con.sql("""SELECT COUNT(index) AS Amount FROM gym_members
                        WHERE WO_Type = 'Strength'""").to_df()


#%%
data_=copy.copy(data)

gender_binary=[]
Workout_Types=[]

#Encoding categorical attributes into numbers
for i in range(len(data_)):
    if data_['Gender'][i] == 'Male':
        gender_binary.append(-1)
    else:
        gender_binary.append(1)
        
for i in range(len(data_)):
    if data_['WO_Type'][i] == 'Yoga':
        Workout_Types.append(0)
    elif data_['WO_Type'][i] == 'HIIT':
        Workout_Types.append(1)
    elif data_['WO_Type'][i] == 'Cardio':
        Workout_Types.append(2)
    else:
        Workout_Types.append(3)
        
data_['Gender']=gender_binary
data_['WO_Type']=Workout_Types

# Correlation matrix

# Select only numerical columns
numeric_df = data_.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

# Visualizing the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.savefig('plots/Correlation_Matrix.png')
plt.show()


#%% Pair Plot

# Create a pairplot graph
pairplot = sns.pairplot(
    data[['Weight (kg)', 'Height (m)', 'BMI', 'Calories_Burned', 'Session_Duration (hours)', 'Gender']],
    hue='Gender',
    palette='coolwarm'
)

# Delete the legend to add a new one and adjust it's fontsize
pairplot._legend.remove()

handles = pairplot._legend_data.values()
labels = pairplot._legend_data.keys()

pairplot.fig.legend(
    handles=handles,
    labels=labels,
    loc='upper center',  
    title='Gender',
    fontsize=12,        
    title_fontsize=15
)

for ax in pairplot.axes.flatten(): 
    if ax is not None:
        ax.set_xlabel(ax.get_xlabel(), fontsize=11)
        ax.set_ylabel(ax.get_ylabel(), fontsize=11) 

pairplot.fig.subplots_adjust(top=0.9)

plt.savefig('plots/pair_plot.png')

plt.show()

#%% EDA (exploratory data analysis)

def summary(df):
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values
    summ['Duplicate'] = df.duplicated().sum()
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['avg'] = desc['mean'].values
    summ['std dev'] = desc['std'].values
    summ['top value'] = desc['top'].values
    summ['Freq'] = desc['freq'].values

    return summ

summary_=summary(data)

summary_.to_csv('plots/Attributes_general_characteristics.csv')

summary_
#%%

#Define a function to easily make bar graphs with diferent variables and column labels.
def bar (column_labels, variable, bin_width, bin_min, bin_max, title, xlabel, ylabel, maxy, colors):
    #Set the plot size
    fig, ax = plt.subplots(figsize=(19.2,10.8))

    # Divide into subsets
    if column_labels == 'Gender':
        obsFemale = data[data[column_labels]=='Female'][variable]
        obsMale = data[data[column_labels]=='Male'][variable]
    
    elif column_labels == 'WO_Type':
        obsYoga = data[data[column_labels]=='Yoga'][variable]
        obsHIIT = data[data[column_labels]=='HIIT'][variable]
        obsCardio = data[data[column_labels]=='Cardio'][variable]
        obsStr = data[data[column_labels]=='Strength'][variable]
    
    else:
        raise ValueError('column label was not found. Please try Gender or WO_Type')

    width = bin_width
    bins = np.arange(bin_min,bin_max, width)
    
    if column_labels == 'Gender':
        
        countsFemale, bins = np.histogram(obsFemale, bins=bins)
        countsMale  , bins = np.histogram(obsMale  , bins=bins)

        freqFemale = countsFemale / float(countsFemale.sum())
        freqMale   = countsMale   / float(countsMale.sum())

        center = (bins[:-1] + bins[1:]) / 2

        # Plot Female
        ax.bar(x=center-width*0.2,        
           height=freqFemale, 
           width=width*.4,       
           align='center',     
           color=colors[0],    
           edgecolor='black')

        # Plot Male
        ax.bar(x=center+width*0.2,    
           height=freqMale,  
           width=width*.4,   
           align='center',     
           color=colors[1],    
           edgecolor='black')  

        ax.set_title(title, fontsize=28)
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.set_ylim(0, maxy)

        
        if '.' in str(bin_width):
            res = len(str(bin_width).split('.')[1])
        else:
            res = 0

        labels = [f'({bins[i]:.{res}f},{bins[i+1]:.{res}f}]' for i in range(len(bins)-1)]
        ax.set_xticks(center)
        ax.set_xticklabels(labels, rotation=30, fontsize=12)
        ax.tick_params(axis='y', labelsize=25)
        ax.tick_params(bottom = False)

        ax.legend(['Female', 'Male'], loc='upper left', fontsize=18)

        plt.grid()
    
    elif column_labels == 'WO_Type':

        countsYoga, bins = np.histogram(obsYoga, bins=bins)   
        countsHIIT  , bins = np.histogram(obsHIIT  , bins=bins)     
        countsCardio, bins = np.histogram(obsCardio, bins=bins)   
        countsStr  , bins = np.histogram(obsStr  , bins=bins)    

        freqYoga = countsYoga / float(countsYoga.sum())
        freqHIIT   = countsHIIT   / float(countsHIIT.sum())
        freqCardio = countsCardio / float(countsCardio.sum())
        freqStr   = countsStr   / float(countsStr.sum())

        center = (bins[:-1] + bins[1:]) / 2                    
        
        if len(colors) != 4:
            raise ValueError("There needs to be 4 colors for WO_Type.")
        
        # Plot Yoga
        ax.bar(x=center-width*0.3,      
           height=freqYoga, 
           width=width*.2,     
           align='center',    
           color=colors[0],   
           edgecolor='black')  

        # Plot HIIT
        ax.bar(x=center-width*0.1,       
           height=freqHIIT,  
           width=width*.2,     
           align='center',    
           color=colors[1],    
           edgecolor='black')   
        
        # Plot Cardio
        ax.bar(x=center+width*0.1,       
           height=freqCardio, 
           width=width*.2,        
           align='center',    
           color=colors[2],  
           edgecolor='black')

        # Plot Strength
        ax.bar(x=center+width*0.3,      
           height=freqStr, 
           width=width*.2,     
           align='center',   
           color=colors[3],   
           edgecolor='black') 

        ax.set_title(title, fontsize=28)
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.set_ylim(0, maxy)
        
        if '.' in str(bin_width):
            res = len(str(bin_width).split('.')[1])
        else:
            res = 0
        
        labels = [f'({bins[i]:.{res}f},{bins[i+1]:.{res}f}]'
               for i in range(len(bins)-1)]  

        ax.set_xticks(center)                       
        ax.set_xticklabels(labels, rotation=30, fontsize=12)
        ax.tick_params(axis='y', labelsize=25)
        ax.tick_params(bottom = False) 

        ax.legend(['Yoga', 'HIIT', 'Cardio', 'Strength'], loc='upper left', fontsize=18)
                
        plt.grid()
    plt.savefig(f'plots/bar/{title} by {column_labels}.png')

#%%
# Age distribution/gender
bar('Gender', 'Age', 5, 15, 60, 'Age distribution', 'Age', 'Relative frequency of the amnt of people', 0.25, ['forestgreen', 'darkslateblue'])
# Height distribution/gender
bar('Gender', 'Height (m)', 0.05, 1.4, 2.10, 'Height distribution', 'Height (m)', 'Relative frequency of the amnt of people', 0.25, ['forestgreen', 'darkslateblue'])
# Weight distribution/gender
bar('Gender', 'Weight (kg)', 5, 20, 150, 'Weight distribution', 'Weight (kg)', 'Relative frequency of the amnt of people', 0.25, ['forestgreen', 'darkslateblue'])
# Max_BPM distribution/gender
bar('Gender', 'Max_BPM', 5, 155, 210, 'Max_BPM distribution', 'Max_BPM', 'Relative frequency of the amnt of people', 0.16, ['forestgreen', 'darkslateblue'])
# Resting_BPM distribution/gender
bar('Gender', 'Resting_BPM', 5, 45, 81, 'Resting_BPM distribution', 'Resting_BPM', 'Relative frequency of the amnt of people', 0.25, ['forestgreen', 'darkslateblue'])
# Session_Duration distribution/gender
bar('Gender', 'Session_Duration (hours)', 0.1, 0.4, 2.3, 'Session_Duration (hours) distribution', 'Session_Duration (hours)', 'Relative frequency of the amnt of people', 0.175, ['forestgreen', 'darkslateblue'])
# Fat_Percentage distribution/gender
bar('Gender', 'Fat_Percentage', 2, 6, 40, 'Fat_Percentage distribution', 'Fat_Percentage', 'Relative frequency of the amnt of people', 0.2, ['forestgreen', 'darkslateblue'])
# Water_Intake distribution/gender
bar('Gender', 'Water_Intake (liters)', 0.2, 1.2, 4.2, 'Water_Intake (liters) distribution', 'Water_Intake (liters)', 'Relative frequency of the amnt of people', 0.35, ['forestgreen', 'darkslateblue'])
# WO_Freq distribution/gender
bar('Gender', 'WO_Freq (days/week)', 1, 1, 8, 'WO_Freq (days-week) distribution', 'WO_Freq (days/week)', 'Relative frequency of the amnt of people', 0.4, ['forestgreen', 'darkslateblue'])
# Experience_Level distribution/gender
bar('Gender', 'Experience_Level', 1, 0, 6, 'Experience_Level distribution', 'Experience_Level', 'Relative frequency of the amnt of people', 0.45, ['forestgreen', 'darkslateblue'])
# BMI distribution/gender
bar('Gender', 'BMI', 5, 5, 56, 'BMI distribution', 'BMI', 'Relative frequency of the amnt of people', 0.4, ['forestgreen', 'darkslateblue'])
# Max_BPM distribution/WO_Type
bar('WO_Type', 'Max_BPM', 5, 155, 210, 'Max_BPM distribution', 'Max_BPM', 'Relative frequency of the amnt of people', 0.16, ['red', 'gold', 'darkorange', 'aquamarine'])
# Calories_Burned distribution/WO_Type
bar('WO_Type', 'Calories_Burned', 200, 0, 2000, 'Calories_Burned distribution', 'Calories_Burned', 'Relative frequency of the amnt of people', 0.35, ['red', 'gold', 'darkorange', 'aquamarine'])
# Age distribution/WO_Type
bar('WO_Type', 'Age', 5, 15, 60, 'Age distribution', 'Age', 'Relative frequency of the amnt of people', 0.25, ['red', 'gold', 'darkorange', 'aquamarine'])
# Session_Duration distribution/WO_Type
bar('WO_Type', 'Session_Duration (hours)', 0.1, 0.4, 2.15, 'Session_Duration (hours) distribution', 'Session_Duration (hours)', 'Relative frequency of the amnt of people', 0.175, ['red', 'gold', 'darkorange', 'aquamarine'])
# Water_Intake distribution/WO_Type
bar('WO_Type', 'Water_Intake (liters)', 0.2, 1, 4, 'Water_Intake (liters) distribution', 'Water_Intake (liters)', 'Relative frequency of the amnt of people', 0.25, ['red', 'gold', 'darkorange', 'aquamarine'])
# Resting_BPM distribution/WO_Type
bar('WO_Type', 'Resting_BPM', 5, 45, 81, 'Resting_BPM distribution', 'Resting_BPM', 'Relative frequency of the amnt of people', 0.3, ['red', 'gold', 'darkorange', 'aquamarine'])

#%% Joint Plot

def joint_plots(data1, data2, column_labels):
    if column_labels == 'Gender':
        data1Female = data[data[column_labels] == 'Female'][data1]
        data1Male = data[data[column_labels] == 'Male'][data1]
        data2Female = data[data[column_labels] == 'Female'][data2]
        data2Male = data[data[column_labels] == 'Male'][data2]

        # Female plot
        sns_plot = sns.jointplot(x=data1Female, y=data2Female, kind='reg', color='red')
        sns_plot.fig.set_size_inches(19.2, 10.8)  
        sns_plot.fig.suptitle(f'{data1} vs. {data2} (Female)', y=0.99, fontsize=27)
        sns_plot.ax_joint.set_xlabel(data1, fontsize=20) 
        sns_plot.ax_joint.set_ylabel(data2, fontsize=20)  
        sns_plot.ax_joint.tick_params(axis='x', labelsize=20)
        sns_plot.ax_joint.tick_params(axis='y', labelsize=20)
        sns_plot.fig.subplots_adjust(top=0.9) 
        plt.grid()
        plt.savefig(f'plots/joint_plots/{data1} vs. {data2} (Female).png')
        plt.show()

        # Male plot
        sns_plot_ = sns.jointplot(x=data1Male, y=data2Male, kind='reg', color='blue')
        sns_plot_.fig.set_size_inches(19.2, 10.8) 
        sns_plot_.fig.suptitle(f'{data1} vs. {data2} (Male)', y=0.99, fontsize=27)
        sns_plot_.ax_joint.set_xlabel(data1, fontsize=20)
        sns_plot_.ax_joint.set_ylabel(data2, fontsize=20) 
        sns_plot_.ax_joint.tick_params(axis='x', labelsize=20)
        sns_plot_.ax_joint.tick_params(axis='y', labelsize=20)
        sns_plot_.fig.subplots_adjust(top=0.9)
        plt.grid()
        plt.savefig(f'plots/joint_plots/{data1} vs. {data2} (Male).png')
        plt.show()

    else:
        plt.figure(figsize=(19.2, 10.8))
        plt.plot(data[data1], data[data2], '.')
        plt.xlabel(data1)
        plt.ylabel(data2)
        plt.xlabel(data1, fontsize=16)  
        plt.ylabel(data2, fontsize=16)
        plt.grid()
        plt.title(f'plots/joint_plots/{data1} vs. {data2}', fontsize=20)
        plt.show()

joint_plots('Weight (kg)', 'Resting_BPM', 'Gender')

joint_plots('Session_Duration (hours)', 'Calories_Burned', 'Gender')

joint_plots('Weight (kg)', 'BMI', 'Gender')

#%% TensorFlow for Calories_Burned

#First, I make a copy of the original data so I don't end up modifying it.
data_Tensor=copy.copy(data)

#Drop unnecessesary columns
data_Tensor = data_Tensor.drop(['index'], axis=1)

#Split the columns into categorical and numerical
categorical_cols = ['Gender', 'WO_Type']

numerical_features = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Session_Duration (hours)', 'Water_Intake (liters)', 
                      'Calories_Burned', 'Fat_Percentage', 'WO_Freq (days/week)', 'Experience_Level', 'BMI']

for col in data_Tensor.columns:
    if col not in numerical_features and col not in categorical_cols:
        data_Tensor = data_Tensor.drop([col], axis=1)

#Scale the numerical features
scaler = StandardScaler()

scaler.fit(data_Tensor[numerical_features])

data_Tensor[numerical_features] = scaler.transform(data_Tensor[numerical_features])

#Take out outliers to improve the fitting of the model
model = IsolationForest(contamination=0.05, random_state=42)
data_Tensor['outlier'] = model.fit_predict(data_Tensor[numerical_features])
data_Tensor = data_Tensor[data_Tensor['outlier'] == 1]
 
#Transform categorical attributes into numbers so they can also be used in the training of the model.
encoder = OrdinalEncoder()
for category in categorical_cols:
    data_Tensor[category] = encoder.fit_transform(data_Tensor[category].values.reshape(-1, 1))

X = data_Tensor.drop(['Calories_Burned', 'outlier'], axis=1).values  
y = data_Tensor['Calories_Burned'].values  

X = X.astype('float')
y = y.astype('float')

#Split the database into train (80%) and Hold (20%)
X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.20, random_state=1)

#The model
model = Sequential()
model.add(Dense(13, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(64, activation='relu')) 
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(1, activation='linear'))  # Regression output layer with linear activation

model.compile(loss='mse', optimizer='adam')
early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

#Model fitting
model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=2, validation_split=0.2, callbacks=[early_stopping])

#Evaluating the model
yhat = model.predict(X_hold)
error = mean_absolute_error(y_hold, yhat)
print('MAE: %.3f' % error)

#Rescale the data back to normal
y_hold = y_hold*scaler.scale_[4]+scaler.mean_[4]
yhat = np.squeeze(yhat*scaler.scale_[4]+scaler.mean_[4])

sorted_pairs = sorted(zip(y_hold, yhat), key=lambda pair: pair[0])

y_hold, yhat = zip(*sorted_pairs)

y_hold = np.array(y_hold)
yhat = np.array(yhat)

# Fit a linear regression
a, b = np.polyfit(y_hold, yhat, deg=1)
y_est = a * y_hold + b  

residuals = yhat - y_est
se_residual = np.sqrt(np.sum(residuals**2) / (len(y_hold) - 2))

# Calculating the condidence band (95%)
n = len(y_hold)
z_value = 1.96
y_err = z_value * y_hold.std() * np.sqrt(1/len(y_hold) + (y_hold - y_hold.mean())**2 / np.sum((y_hold - y_hold.mean())**2))

#Plotting
plt.figure(figsize=(7, 7))
plt.scatter(y_hold, yhat, color='darkturquoise', label='Predicted vs Actual')
plt.plot([min(y_hold), max(y_hold)], [min(y_hold), max(y_hold)], color='red', label='Ideal Prediction (y=x)')
plt.xlabel('Actual Calories_Burned', fontsize=15)
plt.ylabel('Predicted Calories_Burned', fontsize=15)
plt.title('Actual vs Predicted Calories_Burned', fontsize=20)

plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)

plt.plot(y_hold, y_est, '-', color='limegreen', label="Regression Line")
plt.fill_between(y_hold, y_est - y_err, y_est + y_err, alpha=0.2, color='blue', label="95% Confidence Band")

plt.legend()
plt.grid()
plt.savefig('plots/Actual vs Predicted Calories_Burned.png')
plt.show()

mse = mean_squared_error(y_hold, yhat)
rmse = np.sqrt(mse)
r2 = r2_score(y_hold, yhat)

print(f'RMSE: {rmse:.3f}')
print(f'Relative RMSE: {rmse/np.mean(y_hold):.3f}')
print(f'R-squared: {r2:.3f}')





























