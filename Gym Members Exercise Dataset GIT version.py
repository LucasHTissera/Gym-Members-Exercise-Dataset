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

diferent_workouts = con.sql("SELECT DISTINCT WO_Type FROM gym_members").to_df()

amnt_of_males = con.sql("""SELECT COUNT(index) AS Amount FROM gym_members
                        WHERE Gender = 'Male'""").to_df()

amnt_of_females = con.sql("""SELECT COUNT(index) AS Amount FROM gym_members
                        WHERE Gender = 'Female'""").to_df()
                        
#%%
data_=copy.copy(data)

gender_binary=[]
Workout_Types=[]

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

# Matriz de correlacion

numeric_df = data_.select_dtypes(include=[np.number])  # Select only numerical columns
correlation_matrix = numeric_df.corr()

# Visualizing the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


#%% Pair Plot

# Crear el gráfico de pairplot
pairplot = sns.pairplot(
    data[['Weight (kg)', 'Height (m)', 'BMI', 'Calories_Burned', 'Session_Duration (hours)', 'Gender']],
    hue='Gender',
    palette='coolwarm'
)

# Eliminar la leyenda predeterminada
pairplot._legend.remove()

# Crear manualmente la leyenda en la figura completa
handles = pairplot._legend_data.values()
labels = pairplot._legend_data.keys()

# Añadir la leyenda al nivel de la figura
pairplot.fig.legend(
    handles=handles,
    labels=labels,
    loc='upper center',  # Ajusta la posición relativa a toda la figura
    title='Gender',
    fontsize=12,         # Cambiar el tamaño de las etiquetas
    title_fontsize=15    # Cambiar el tamaño del título
)

for ax in pairplot.axes.flatten():  # Iterar sobre todos los ejes del pairplot
    if ax is not None:  # Algunos ejes podrían ser `None`
        ax.set_xlabel(ax.get_xlabel(), fontsize=11)  # Cambia el tamaño del texto en el eje X
        ax.set_ylabel(ax.get_ylabel(), fontsize=11)  # Cambia el tamaño del texto en el eje Y


# Ajustar los márgenes para que la leyenda no se sobreponga
pairplot.fig.subplots_adjust(top=0.9)

plt.savefig('pair_plot.png')

plt.show()

#%% EDA
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

summary_.to_csv('summary.csv')

summary_
#%%

def barras (column_labels, variable, bin_width, bin_min, bin_max, title, xlabel, ylabel, maxy, colors):
    fig, ax = plt.subplots(figsize=(19.2,10.8))

    # Armamos dos subsets: Male y Female
    
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

    # Calculamos datos necesarios para generar las barras
    # bins ...
    width = bin_width
    bins = np.arange(bin_min,bin_max, width)
    
    if column_labels == 'Gender':
        
        # Contamos cuantos de los datos caen en cada uno de los bins
        countsFemale, bins = np.histogram(obsFemale, bins=bins)     # Hace el histograma (por bin)
        countsMale  , bins = np.histogram(obsMale  , bins=bins)     # Hace el histograma (por bin)
        # Si queremos graficar la frecuencia en vez de la cantidad, la calculamos
        freqFemale = countsFemale / float(countsFemale.sum())
        freqMale   = countsMale   / float(countsMale.sum())

        # Fijamos la ubicacion de cada bin
        center = (bins[:-1] + bins[1:]) / 2                         # Calcula el centro de cada barra

        # Graficamos Female
        ax.bar(x=center-width*0.2,        # Ubicacion en el eje x de cada bin
           height=freqFemale, # Alto de la barra
           width=width*.4,         # Ancho de la barra
           align='center',      # Barra centrada
           color=colors[0],     # Color de la barra
           edgecolor='black')   # Color del borde de la barra

        # Graficamos Male
        ax.bar(x=center+width*0.2,        # Ubicacion en el eje x de cada bin
           height=freqMale,   # Alto de la barra
           width=width*.4,      # Ancho de la barra
           align='center',      # Barra centrada
           color=colors[1],     # Color de la barra
           edgecolor='black')   # Color del borde de la barra

        # Agrega titulo, etiquetas a los ejes y limita el rango de valores de los ejes    
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_ylim(0, maxy)

        
        #Resolution is assigned depending on the decimal spaces of bin_width
        if '.' in str(bin_width):
            res = len(str(bin_width).split('.')[1]) #resolution
        else:
            res = 0

        labels = [f'({bins[i]:.{res}f},{bins[i+1]:.{res}f}]' for i in range(len(bins)-1)] # Genera el string de los labels del estilo (v1, v2]

        ax.set_xticks(center)                        # Ubica los ticks del eje x
        ax.set_xticklabels(labels, rotation=60, fontsize=10) # Asigna labels a los ticks del eje x
        ax.tick_params(bottom = False)                       # Remueve los ticks del eje x

        #Agrega leyenda
        ax.legend(['Femenino', 'Masculino'], loc='upper left')

        plt.grid()
    
    elif column_labels == 'WO_Type':
        # Contamos cuantos de los datos caen en cada uno de los bins
        countsYoga, bins = np.histogram(obsYoga, bins=bins)     # Hace el histograma (por bin)
        countsHIIT  , bins = np.histogram(obsHIIT  , bins=bins)     # Hace el histograma (por bin)
        countsCardio, bins = np.histogram(obsCardio, bins=bins)     # Hace el histograma (por bin)
        countsStr  , bins = np.histogram(obsStr  , bins=bins)     # Hace el histograma (por bin)
        # Si queremos graficar la frecuencia en vez de la cantidad, la calculamos
        freqYoga = countsYoga / float(countsYoga.sum())
        freqHIIT   = countsHIIT   / float(countsHIIT.sum())
        freqCardio = countsCardio / float(countsCardio.sum())
        freqStr   = countsStr   / float(countsStr.sum())

        # Fijamos la ubicacion de cada bin
        center = (bins[:-1] + bins[1:]) / 2                         # Calcula el centro de cada barra
        
        if len(colors) != 4:
            raise ValueError("There needs to be 4 colors for WO_Type.")
        
        # Graficamos Yoga
        ax.bar(x=center-width*0.4,        # Ubicacion en el eje x de cada bin
           height=freqYoga, # Alto de la barra
           width=width*.2,         # Ancho de la barra
           align='center',      # Barra centrada
           color=colors[0],     # Color de la barra
           edgecolor='black')   # Color del borde de la barra

        # Graficamos HIIT
        ax.bar(x=center+width*0.2,        # Ubicacion en el eje x de cada bin
           height=freqHIIT,   # Alto de la barra
           width=width*.2,      # Ancho de la barra
           align='center',      # Barra centrada
           color=colors[1],     # Color de la barra
           edgecolor='black')   # Color del borde de la barra
        
        # Graficamos Cardio
        ax.bar(x=center-width*0.2,        # Ubicacion en el eje x de cada bin
           height=freqCardio, # Alto de la barra
           width=width*.2,         # Ancho de la barra
           align='center',      # Barra centrada
           color=colors[2],     # Color de la barra
           edgecolor='black')   # Color del borde de la barra

        # Graficamos Strength
        ax.bar(x=center+width*0.4,        # Ubicacion en el eje x de cada bin
           height=freqStr,   # Alto de la barra
           width=width*.2,      # Ancho de la barra
           align='center',      # Barra centrada
           color=colors[3],     # Color de la barra
           edgecolor='black')   # Color del borde de la barra

        # Agrega titulo, etiquetas a los ejes y limita el rango de valores de los ejes    
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_ylim(0, maxy)
        
        if '.' in str(bin_width):
            res = len(str(bin_width).split('.')[1]) #resolution
        else:
            res = 0
        
        labels = [f'({bins[i]:.{res}f},{bins[i+1]:.{res}f}]'
               for i in range(len(bins)-1)]  # Genera el string de los labels del estilo (v1, v2]

        ax.set_xticks(center)                        # Ubica los ticks del eje x
        ax.set_xticklabels(labels, rotation=30, fontsize=12) # Asigna labels a los ticks del eje x
        ax.tick_params(bottom = False)                       # Remueve los ticks del eje x

        #Agrega leyenda
        ax.legend(['Yoga', 'HIIT', 'Cardio', 'Strength'], loc='upper left')
                
        plt.grid()
    plt.savefig(f'{title} by {column_labels}.png')

#%%
# # Age distribution
barras('Gender', 'Age', 5, 15, 60, 'Age distribution', 'Age', 'Relative frequency of the amnt of people', 0.25, ['forestgreen', 'darkslateblue'])

barras('Gender', 'Height (m)', 0.05, 1.4, 2.10, 'Height distribution', 'Height (m)', 'Relative frequency of the amnt of people', 0.25, ['forestgreen', 'darkslateblue'])

#Es interesante en este grafico como mujeres tienen una distribucion mas chica de peso, en comparacion con hombres.
#Esto podria indicar que mujeres de mayor peso por alguna razon no van al gymnacio, por ejemplo, por miedo a ser juzgadas.
barras('Gender', 'Weight (kg)', 5, 20, 150, 'Weight distribution', 'Weight (m)', 'Relative frequency of the amnt of people', 0.25, ['forestgreen', 'darkslateblue'])

barras('Gender', 'Max_BPM', 5, 155, 210, 'Max_BPM distribution', 'Max_BPM', 'Relative frequency of the amnt of people', 0.16, ['forestgreen', 'darkslateblue'])

barras('Gender', 'Resting_BPM', 5, 45, 81, 'Resting_BPM distribution', 'Resting_BPM', 'Relative frequency of the amnt of people', 0.25, ['forestgreen', 'darkslateblue'])

#En este grafico, se ve q hay 3 duraciones distinguidas, independiente de genero
barras('Gender', 'Session_Duration (hours)', 0.1, 0.4, 2.3, 'Session_Duration (hours) distribution', 'Session_Duration (hours)', 'Relative frequency of the amnt of people', 0.175, ['forestgreen', 'darkslateblue'])

#En este grafico, hay 2 regiones distinguidas en hombres y mujeres, pero en hombres es mas bajo.
barras('Gender', 'Fat_Percentage', 1, 5, 40, 'Fat_Percentage distribution', 'Fat_Percentage', 'Relative frequency of the amnt of people', 0.1, ['forestgreen', 'darkslateblue'])

#En este grafico, se ve q mujeres toman menos agua en promedio q hombres. Sin embargo, en ambos generos, hay un pico. Puede deberse a un ejercicio en particular.
barras('Gender', 'Water_Intake (liters)', 0.2, 1.2, 4.2, 'Water_Intake (liters) distribution', 'Water_Intake (liters)', 'Relative frequency of the amnt of people', 0.35, ['forestgreen', 'darkslateblue'])

barras('Gender', 'WO_Freq (days/week)', 1, 1, 8, 'WO_Freq (days/week) distribution', 'WO_Freq (days/week)', 'Relative frequency of the amnt of people', 0.4, ['forestgreen', 'darkslateblue'])

#Los niveles de experiencia son indistinguibles entre los generos.
barras('Gender', 'Experience_Level', 1, 0, 6, 'Experience_Level distribution', 'Experience_Level', 'Relative frequency of the amnt of people', 0.45, ['forestgreen', 'darkslateblue'])

#El BMI de mujeres es en promedio menor al de hombres.
barras('Gender', 'BMI', 5, 5, 56, 'BMI distribution', 'BMI', 'Relative frequency of the amnt of people', 0.4, ['forestgreen', 'darkslateblue'])

barras('WO_Type', 'Max_BPM', 5, 155, 210, 'Max_BPM distribution', 'Max_BPM', 'Relative frequency of the amnt of people', 0.16, ['red', 'gold', 'darkorange', 'aquamarine'])

barras('WO_Type', 'Calories_Burned', 200, 0, 2000, 'Calories_Burned distribution', 'Calories_Burned', 'Relative frequency of the amnt of people', 0.35, ['red', 'gold', 'darkorange', 'aquamarine'])

barras('WO_Type', 'Age', 5, 15, 60, 'Age distribution', 'Age', 'Relative frequency of the amnt of people', 0.25, ['red', 'gold', 'darkorange', 'aquamarine'])

barras('WO_Type', 'Session_Duration (hours)', 0.1, 0.4, 2.15, 'Session_Duration (hours) distribution', 'Session_Duration (hours)', 'Relative frequency of the amnt of people', 0.175, ['red', 'gold', 'darkorange', 'aquamarine'])

barras('WO_Type', 'Water_Intake (liters)', 0.2, 1, 4, 'Water_Intake (liters) distribution', 'Water_Intake (liters)', 'Relative frequency of the amnt of people', 0.35, ['red', 'gold', 'darkorange', 'aquamarine'])

barras('WO_Type', 'Resting_BPM', 5, 45, 81, 'Resting_BPM distribution', 'Resting_BPM', 'Relative frequency of the amnt of people', 0.3, ['red', 'gold', 'darkorange', 'aquamarine'])


#%% Joint Plot

def joint_plots(data1, data2, column_labels):
    if column_labels == 'Gender':
        data1Female = data[data[column_labels] == 'Female'][data1]
        data1Male = data[data[column_labels] == 'Male'][data1]
        data2Female = data[data[column_labels] == 'Female'][data2]
        data2Male = data[data[column_labels] == 'Male'][data2]

        # Gráfico para Female
        sns_plot = sns.jointplot(x=data1Female, y=data2Female, kind='reg', color='red')
        sns_plot.fig.set_size_inches(19.2, 10.8)  # Ajusta el tamaño a 1920x1080 píxeles
        sns_plot.fig.suptitle(f'{data1} vs. {data2} (Female)', y=0.99, fontsize=20)
        sns_plot.ax_joint.set_xlabel(data1, fontsize=16)  # Etiqueta x con tamaño
        sns_plot.ax_joint.set_ylabel(data2, fontsize=16)  # Etiqueta y con tamaño
        sns_plot.fig.subplots_adjust(top=0.9)  # Ajusta el espacio superior
        plt.savefig(f'{data1} vs. {data2} (Female).png')
        plt.show()

        # Gráfico para Male
        sns_plot_ = sns.jointplot(x=data1Male, y=data2Male, kind='reg', color='blue')
        sns_plot_.fig.set_size_inches(19.2, 10.8)  # Ajusta el tamaño a 1920x1080 píxeles
        sns_plot_.fig.suptitle(f'{data1} vs. {data2} (Male)', y=0.99, fontsize=20)
        sns_plot_.ax_joint.set_xlabel(data1, fontsize=16)  # Etiqueta x con tamaño
        sns_plot_.ax_joint.set_ylabel(data2, fontsize=16)  # Etiqueta y con tamaño
        sns_plot_.fig.subplots_adjust(top=0.9)  # Ajusta el espacio superior
        
        plt.savefig(f'{data1} vs. {data2} (Male).png')
        plt.show()

    else:
        plt.figure(figsize=(19.2, 10.8))
        plt.plot(data[data1], data[data2], '.')
        plt.xlabel(data1)
        plt.ylabel(data2)
        plt.xlabel(data1, fontsize=16)  # Etiqueta x con tamaño
        plt.ylabel(data2, fontsize=16)  # Etiqueta y con tamaño
        plt.title(f'{data1} vs. {data2}', fontsize=20)
        plt.show()
    plt.savefig(f'{data1} vs. {data2} ({column_labels}).png')

joint_plots('Weight (kg)', 'Resting_BPM', 'Gender')

joint_plots('Session_Duration (hours)', 'Calories_Burned', 'Gender')

joint_plots('Weight (kg)', 'BMI', 'Gender')

#%% TensorFlow para Calories_Burned

data_Tensor=copy.copy(data)

data_Tensor = data_Tensor.drop(['index'], axis=1)

categorical_cols = ['Gender']

numerical_features = ['Weight (kg)', 'Height (m)', 'Session_Duration (hours)', 'Water_Intake (liters)', 'Calories_Burned',
                      'Fat_Percentage', 'Water_Intake (liters)', 'WO_Freq (days/week)', 'Experience_Level', 'BMI']

for col in data_Tensor.columns:
    if col not in numerical_features and col not in categorical_cols:
        data_Tensor = data_Tensor.drop([col], axis=1)

scaler = StandardScaler()

scaler.fit(data_Tensor[numerical_features])

data_Tensor[numerical_features] = scaler.transform(data_Tensor[numerical_features])

model = IsolationForest(contamination=0.05, random_state=42)
data_Tensor['outlier'] = model.fit_predict(data_Tensor[numerical_features])
data_Tensor = data_Tensor[data_Tensor['outlier'] == 1]

encoder = OrdinalEncoder()
for category in categorical_cols:
    data_Tensor[category] = encoder.fit_transform(data_Tensor[category].values.reshape(-1, 1))

X = data_Tensor.drop(['Calories_Burned'], axis=1).values  
y = data_Tensor['Calories_Burned'].values  

X = X.astype('float')
y = y.astype('float')

X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.20, random_state=1)

model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(tf.keras.layers.Dropout(0.05))
model.add(Dense(1, activation='linear'))  # Regression output layer with linear activation

model.compile(loss='mse', optimizer='adam')
early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=2, validation_split=0.2, callbacks=[early_stopping])

yhat = model.predict(X_hold)
error = mean_absolute_error(y_hold, yhat)
print('MAE: %.3f' % error)

y_hold = y_hold*scaler.scale_[4]+scaler.mean_[4]
yhat = yhat*scaler.scale_[4]+scaler.mean_[4]

plt.figure(figsize=(10, 6))
plt.scatter(y_hold, yhat, color='blue', label='Predicted vs Actual')
plt.plot([min(y_hold), max(y_hold)], [min(y_hold), max(y_hold)], color='red', label='Ideal Prediction (y=x)')
plt.xlabel('Actual Calories_Burned')
plt.ylabel('Predicted Calories_Burned')
plt.title('Actual vs Predicted Calories_Burned')
plt.legend()
plt.show()

mse = mean_squared_error(y_hold, yhat)
rmse = np.sqrt(mse)
r2 = r2_score(y_hold, yhat)

print(f'MSE: {mse:.3f}')
print(f'RMSE: {rmse:.3f}')
print(f'R-squared: {r2:.3f}')





























