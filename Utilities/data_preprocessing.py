import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

def value_summary(df, sort='null_num', dtype='numeric'):
    """
    compute value stats for each columns
    
    df: DataFrame
    sort: string
    """
    if dtype == 'numeric':
    
        temp_df = pd.DataFrame(
            {
                'null_num': df.isnull().sum(),
                'null_rate': df.isnull().sum()/df.shape[0],
                'unique_vals': df.nunique(), 
                'min': df.min(),
                'qt_25': df.quantile(q=0.25),
                'median': df.quantile(q=0.5),
                'qt_75': df.quantile(q=0.75),
                'max': df.max(),
                'skew': df.skew(axis=0)
            },
            index = df.columns
        ).sort_values(sort)
        
    elif dtype == 'categorical':
        
        temp_df = pd.DataFrame(
            {
                'null_num': df.isnull().sum(),
                'null_rate': df.isnull().sum()/df.shape[0],
                'unique_vals': df.nunique()
            },
            index = df.columns
        ).sort_values(sort)
    
    else:
        raise ValueError('wrong dtype!')
    
    return temp_df

def data_cleanse(df):
    """
    remove columns with single value, sparse catagorical values 
    or related to hardship from dataframe
    
    df: DataFrame
    """
    
    #columns with same value(including NA)
    same_col = df.columns[df.nunique(dropna=False) == 1].tolist()
    
    #columns with high null value rate
    null_col = df.columns[df.isnull().sum()/df.shape[0] >= 0.9].tolist()
    
    #columns related to hardship
    hard_col = [col for col in df.columns if ('hardship' in col) or ('settlement' in col) or ('deferral' in col) or ('recover' in col)]
    
    #columns with sparse catagorical value
    sparse_col = ['url', 'zip_code', 'title', 'emp_title', 'desc', 'sub_grade']
    
    cols = set(same_col + null_col + hard_col + sparse_col)
    df = df.drop(columns=cols)
    
    return df

def date_to_month(date):
    """
    compute difference by month between given date to Jan-2000
    
    date: Series of string, in format '%b-%Y'
    """
    
    t = pd.to_datetime(date, format ='%b-%Y')
    year, month = t.dt.year, t.dt.month
    
    return (year-2000)*12. + (month-1)

def states_to_census_regions(state):
    """
    map states to region
    
    state: Series of string, in state code
    """
    
    mapping = pd.read_csv('https://raw.githubusercontent.com/cphalpert/census-regions/master/us%20census%20bureau%20regions%20and%20divisions.csv')
    
    mapping = mapping[['State Code', 'Region']].merge(state, how = 'right', left_on = 'State Code', right_on = state.name)
    
    return mapping['Region'].values
    
def data_preprocessing(df):
    """
    data preprocessing
    
    df: DataFrame
    """
    
    # remove columns of minimum information
    df = data_cleanse(df)
    
    # convert date variables to integer in months
    df['issue_d_mths'] = date_to_month(df['issue_d'])
    df['last_pymnt_d_mths'] = date_to_month(df['last_pymnt_d'])
    df['earliest_cr_line_mths'] = date_to_month(df['earliest_cr_line'])
    df['last_credit_pull_d_mths'] = date_to_month(df['last_credit_pull_d'])
    
    df = df.drop(columns=['earliest_cr_line', 'issue_d', 'last_credit_pull_d', 'last_pymnt_d'])
    
    #convert state to census region
    df['addr_region'] = states_to_census_regions(df['addr_state'])
    
    df = df.drop(columns=['addr_state'])
    
    return df

def feature_engineering(X_train, X_test, y_train, y_test):
    """
    feature engineering
    
    df: DataFrame
    """
    X_num = X_train.select_dtypes(include='float64').astype('float32')
    X_cat = X_train.select_dtypes(include='object').astype('category')
    
    # catagorical imputation
    cat_imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='missing')
    data_cat = cat_imp.fit_transform(X_cat)
    X_cat = pd.DataFrame(data_cat, columns=X_cat.columns, index=X_cat.index)
    
    # ordinal encoding
    ord_cols = ['term', 'grade', 'emp_length']
    
    term_cat = ['missing', ' 36 months', ' 60 months']
    grade_cat = ['missing', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    emp_cat = ['missing', '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']

    ode = OrdinalEncoder(categories=[term_cat, grade_cat, emp_cat], dtype=np.int8)
    data_ode = ode.fit_transform(X_cat[ord_cols])
    df_ode = pd.DataFrame(data_ode, columns=ord_cols, index=X_cat.index)
    
    # onehot encoding
    oh_cols = ['home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type', 'disbursement_method', 'addr_region']
    ohe = OneHotEncoder(dtype=np.int8)
    data_ohe = ohe.fit_transform(X_cat[oh_cols]).toarray()
    df_ohe = pd.DataFrame(data_ohe, columns=ohe.get_feature_names(oh_cols), index=X_cat.index)
    
    # numeric abnormal indicators 1 for dominated 2 for missing 0 for others
    mod_cols = X_num.columns[X_num.apply(lambda x: x.value_counts(normalize=True).iloc[0,]) >= 0.5]
    mis_cols = X_num.columns[X_num.isnull().mean() >= 0.5]
    diff_mod = [col for col in mis_cols if col not in mod_cols]
    diff_mis = [col for col in mod_cols if col not in mis_cols]
    col_mode = X_num[mod_cols].mode().iloc[0]
    
    mod_temp = (X_num[mod_cols] == col_mode).astype(np.int8)
    mis_temp = X_num[mis_cols].isnull().astype(np.int8)
    
    mod_temp[diff_mod] = pd.DataFrame([[0 for _ in range(len(diff_mod))]], index=mod_temp.index, dtype=np.int8)
    mis_temp[diff_mis] = pd.DataFrame([[0 for _ in range(len(diff_mis))]], index=mis_temp.index, dtype=np.int8)
    
    df_abnormal = mod_temp + mis_temp*2
    df_abnormal = df_abnormal.add_suffix('_ind')
    
    # numeric imputation
    lm_cols = X_num.columns[X_num.isnull().mean() < 0.5]
    
    lm_imp = SimpleImputer(missing_values=np.nan, strategy='median')
    data_lm = lm_imp.fit_transform(X_num[lm_cols])
    df_lm = pd.DataFrame(data_lm, columns=lm_cols, index=X_num.index)
    
    gm_imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.)
    data_gm = gm_imp.fit_transform(X_num[mis_cols])
    df_gm = pd.DataFrame(data_gm, columns=mis_cols, index=X_num.index)
    
    df_num = pd.concat([df_lm, df_gm], axis=1)
    
    # numeric skewness
    skw_cols = df_num.columns[np.abs(df_num.skew())>1.]
    skw_cols = [col for col in skw_cols if col not in mod_cols and col not in mis_cols]
    skw_q1 = df_num[skw_cols].quantile(q=0.25)
    skw_q3 = df_num[skw_cols].quantile(q=0.75)
    skw_IQR = skw_q3 - skw_q1
    IQR_lower, IQR_higher = skw_q1 - skw_IQR*1.5, skw_q3 + skw_IQR*1.5

    df_num[skw_cols] = df_num[skw_cols].clip(IQR_lower, IQR_higher, axis=1).astype(np.float32)
    
    # target labelize
    lbe = OrdinalEncoder(categories=[['Fully Paid', 'Charged Off']], dtype=np.int8)
    data_lbe = lbe.fit_transform(y_train)
    y_train = pd.DataFrame(data_lbe, columns=y_train.columns, index=y_train.index)

    X_train = pd.concat([df_ode, df_ohe, df_abnormal, df_num], axis=1)
    
    # test features
    X_num_test = X_test.select_dtypes(include='float64').astype('float32')
    X_cat_test = X_test.select_dtypes(include='object').astype('category')
    
    data_cat_test = cat_imp.transform(X_cat_test)
    X_cat_test = pd.DataFrame(data_cat_test, columns=X_cat_test.columns, index=X_cat_test.index)
    
    data_ode_test = ode.transform(X_cat_test[ord_cols])
    df_ode_test = pd.DataFrame(data_ode_test, columns=ord_cols, index=X_cat_test.index)
    
    data_ohe_test = ohe.transform(X_cat_test[oh_cols]).toarray()
    df_ohe_test = pd.DataFrame(data_ohe_test, columns=ohe.get_feature_names(oh_cols), index=X_cat_test.index)
    
    mod_temp_test = (X_num_test[mod_cols] == col_mode).astype(np.int8)
    mis_temp_test = X_num_test[mis_cols].isnull().astype(np.int8)
    mod_temp_test[diff_mod] = pd.DataFrame([[0 for _ in range(len(diff_mod))]], index=mod_temp_test.index, dtype=np.int8)
    mis_temp_test[diff_mis] = pd.DataFrame([[0 for _ in range(len(diff_mis))]], index=mis_temp_test.index, dtype=np.int8)
    df_abnormal_test = mod_temp_test + mis_temp_test*2
    df_abnormal_test = df_abnormal_test.add_suffix('_ind')
    
    data_lm_test = lm_imp.transform(X_num_test[lm_cols])
    df_lm_test = pd.DataFrame(data_lm_test, columns=lm_cols, index=X_num_test.index)
    
    data_gm_test = gm_imp.transform(X_num_test[mis_cols])
    df_gm_test = pd.DataFrame(data_gm_test, columns=mis_cols, index=X_num_test.index)
    
    df_num_test = pd.concat([df_lm_test, df_gm_test], axis=1)
    
    df_num_test[skw_cols] = df_num_test[skw_cols].clip(IQR_lower, IQR_higher, axis=1).astype(np.float32)
    
    data_lbe_test = lbe.transform(y_test)
    y_test = pd.DataFrame(data_lbe_test, columns=y_test.columns, index=y_test.index)
    
    X_test = pd.concat([df_ode_test, df_ohe_test, df_abnormal_test, df_num_test], axis=1)
    
    return X_train, X_test, y_train, y_test