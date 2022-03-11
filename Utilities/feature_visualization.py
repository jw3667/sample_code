import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_data_processing_cat(df1, df2, var):
    """
    This functions takes dataset prior and after data cleaning, and compare distributional changes of categorical variables
    df1: old dataset
    df2: new dataset
    var: variable interested
    """
    plt.figure(figsize=(15,10))
    plt.subplot(2,2,1)
    var_temp1 = df1[var].drop_duplicates().sort_values().to_list()
    sns.countplot(x=var,data=df1,hue='loan_status',order=var_temp1)
    
    plt.subplot(2,2,2)
    var_temp2 = df2[var].drop_duplicates().sort_values().to_list()
    g=sns.countplot(x=var,data=df2,hue='loan_status',order=var_temp2)
    
    return g


def plot_data_processing_num(df1, df2, var):
    """
    This functions takes dataset prior and after data cleaning, and compare distributional changes of nemeric variables
    df1: old dataset
    df2: new dataset
    var: variable interested
    """
    plt.figure(figsize=(15,10))
    plt.subplot(2,2,1)
    sns.histplot(x=var,data=df1,hue='loan_status',bins=50,stat='probability',common_norm=False)
    
    plt.subplot(2,2,2)
    var_temp2 = df2[var].drop_duplicates().sort_values().to_list()
    g=sns.histplot(x=var,data=df2,hue='loan_status',bins=50,stat='probability',common_norm=False)
    
    return g


def plot_num2(df,var1,var2):
    """
    This functions takes dataset prior and after data cleaning, and compare distributional changes of nemeric variables
    df: input dataset
    var1: variable of interest 1
    var2: variable of interest 2
    """
    plt.figure(figsize=(15,10))
    plt.subplot(2,2,1)
    sns.histplot(x=var1,data=df,hue='loan_status',bins=50,stat='probability',common_norm=False)
    
    plt.subplot(2,2,2)
    var_temp2 = df[var2].drop_duplicates().sort_values().to_list()
    g=sns.histplot(x=var2,data=df,hue='loan_status',bins=50,stat='probability',common_norm=False)
    
    return g


def plot_num3(df,var1,var2,var3):
    """
    This functions takes dataset prior and after data cleaning, and compare distributional changes of nemeric variables
    df: input dataset
    var1: variable of interest 1
    var2: variable of interest 2
    """
    plt.figure(figsize=(15,10))
    plt.subplot(3,2,1)
    sns.histplot(x=var1,data=df,hue='loan_status',bins=50,stat='probability',common_norm=False)
    
    plt.subplot(3,2,2)
    var_temp2 = df[var2].drop_duplicates().sort_values().to_list()
    g=sns.histplot(x=var2,data=df,hue='loan_status',bins=50,stat='probability',common_norm=False)
    
    plt.subplot(3,2,3)
    var_temp3 = df[var3].drop_duplicates().sort_values().to_list()
    g=sns.histplot(x=var3,data=df,hue='loan_status',bins=50,stat='probability',common_norm=False)
    
    return g
