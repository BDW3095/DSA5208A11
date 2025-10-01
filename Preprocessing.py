from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

columns=["tpep_pickup_datetime","tpep_dropoff_datetime","passenger_count","trip_distance",
         "RatecodeID","PULocationID","DOLocationID","payment_type","extra","total_amount"]

file_path=Path("./nytaxi2022.csv")
out_path="./processedData/"
df=pd.read_csv(file_path,usecols=columns)

def preprocess_and_split(df):
    
    np.random.seed(5208)

    numeric_columns=["duration_min","trip_distance","extra"]
    categorical_columns=["passenger_count","RatecodeID","PULocationID","DOLocationID","payment_type","pickup_month","pickup_day","pickup_hour"]

    target_column="total_amount"

    top_pairs=df.groupby(["PULocationID","DOLocationID"]).size().nlargest(500).index
    df=df[df.set_index(["PULocationID","DOLocationID"]).index.isin(top_pairs)].reset_index(drop=True)

    df=df[df["passenger_count"]==1]
    df=df[(df["trip_distance"]>0) & (df["trip_distance"]<=20)]
    df=df[df["RatecodeID"]==1]
    df=df[df["payment_type"]==1]
    df=df[df["extra"]>=0]
    df=df[(df["total_amount"]>0) & (df["total_amount"]<=100)]

    df=df.drop_duplicates(subset=["tpep_pickup_datetime","tpep_dropoff_datetime","PULocationID","DOLocationID","trip_distance","total_amount"])

    df["tpep_pickup_datetime"]=pd.to_datetime(df["tpep_pickup_datetime"],format="mixed",errors="coerce")
    df["tpep_dropoff_datetime"]=pd.to_datetime(df["tpep_dropoff_datetime"],format="mixed",errors="coerce")

    df=df.dropna(subset=["tpep_pickup_datetime","tpep_dropoff_datetime"])

    df["pickup_month"]=df["tpep_pickup_datetime"].dt.month.astype("int8")
    df["pickup_day"]=(df["tpep_pickup_datetime"].dt.dayofweek+1).astype("int8")
    df["pickup_hour"] =df["tpep_pickup_datetime"].dt.hour.astype("int8")
    df["duration_min"]=(df["tpep_dropoff_datetime"]-df["tpep_pickup_datetime"]).dt.total_seconds()/60

    df=df.drop(columns=["tpep_pickup_datetime","tpep_dropoff_datetime"])

    df=df[(df['pickup_month'].between(1,12)) & (df['pickup_day'].between(1,7)) & (df['pickup_hour'].between(0,23))]
    df=df[(df["duration_min"]>0) & (df["duration_min"]<=60)]

    X_categorical=pd.get_dummies(df[categorical_columns].astype(str),dtype="int8")
    X_numeric=df[numeric_columns].astype("float32")

    X=pd.concat([X_numeric,X_categorical],axis=1)
    y=df[target_column].astype("float32")

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=5208)

    scaler=StandardScaler()
    X_train.loc[:,numeric_columns]=scaler.fit_transform(X_train[numeric_columns]).astype("float32")
    X_test.loc[:,numeric_columns]=scaler.transform(X_test[numeric_columns]).astype("float32")

    X_train.to_csv(out_path+"Xtrain.csv",index=False)
    y_train.to_csv(out_path+"ytrain.csv",index=False,header=True)
    X_test.to_csv(out_path+"Xtest.csv",index=False)
    y_test.to_csv(out_path+"ytest.csv",index=False,header=True)

    return X_train,y_train,X_test,y_test

X_train,y_train,X_test,y_test=preprocess_and_split(df)

