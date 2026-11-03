"""Deterministic raw-data loaders; no dependency-dependent AIF360 fallbacks."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import hashlib
import numpy as np
import pandas as pd
import requests
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
DATASETS = ['german', 'heart_disease', 'compas', 'lsac', 'bank_marketing', 'adult', 'folktables_income']
DOMAINS = {'german':'Finance','heart_disease':'Healthcare','compas':'Criminal justice','lsac':'Education','bank_marketing':'Finance','adult':'Census income','folktables_income':'Census income'}
OUTCOMES = {'german':'1=good credit (UCI original class 1); favorable=1', 'heart_disease':'1=any diagnosed disease (num>0); favorable=0', 'compas':'1=two-year recidivism; favorable=0', 'lsac':'1=standardized first-year GPA above cohort median; favorable=1; not bar passage', 'bank_marketing':'1=term-deposit subscription; favorable not established', 'adult':'1=income >USD 50,000; favorable=1', 'folktables_income':'1=personal income >USD 50,000; favorable=1; CA 2018 canonical ACSIncome eligibility'}
SOURCES = {'german':'https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data','heart_disease':'https://archive.ics.uci.edu/dataset/45/heart+disease','compas':'https://github.com/propublica/compas-analysis','lsac':'https://github.com/Trusted-AI/AIF360/tree/master/aif360/data/raw/law_school_gpa','bank_marketing':'https://archive.ics.uci.edu/dataset/222/bank+marketing','adult':'https://www.openml.org/d/1590','folktables_income':'https://github.com/socialfoundations/folktables'}

def download(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=120); r.raise_for_status()
        path.write_bytes(r.content)
    return path

def load(name, age_encoding='age25'):
    metadata = {'dataset':name,'domain':DOMAINS[name],'outcome':OUTCOMES[name], 'source':SOURCES[name]}
    if name == 'german':
        cols = ['checking_status','duration','credit_history','purpose','credit_amount','savings_status','employment','installment_commitment','personal_status','other_parties','residence_since','property_magnitude','age','other_payment_plans','housing','existing_credits','job','num_dependents','own_telephone','foreign_worker','credit']
        path = download('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data', DATA/'german/german.data')
        df = pd.read_csv(path, sep=r'\s+', names=cols)
        y = df.pop('credit').eq(1).astype(int)
        age = df.pop('age')
        if age_encoding == 'raw_age': group = age
        elif age_encoding in ['age25','age35']: group = age.ge(int(age_encoding[3:])).astype(int)
        elif age_encoding == 'age_bands': group = pd.cut(age, [-np.inf,25,40,np.inf], right=False).astype(str)
        else: raise ValueError(age_encoding)
        s = pd.DataFrame({'age':group}); X = df
        metadata['age_encoding'] = age_encoding
    elif name == 'compas':
        path = download('https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv', DATA/'compas/compas-scores-two-years.csv')
        df = pd.read_csv(path)
        df = df.loc[df.days_b_screening_arrest.between(-30,30) & df.is_recid.ne(-1) & df.c_charge_degree.ne('O') & df.score_text.ne('N/A'), ['sex','race','age','c_charge_degree','priors_count','two_year_recid']].copy()
        y = df.pop('two_year_recid').astype(int); s = df[['sex','race']].copy(); X = df.drop(columns=s.columns)
    elif name == 'lsac':
        path = download('https://raw.githubusercontent.com/jkomiyama/fairregresion/master/dataset/law/lsac.sas7bdat', DATA/'revision/lsac.sas7bdat')
        df = pd.read_sas(path,format='sas7bdat',encoding='utf-8')
        df = df.loc[df.race1.isin(['black','white'])].copy()
        df['gender'] = df.gender.fillna('female')
        df = df[['race1','gender','lsat','ugpa','zfygpa']].dropna().reset_index(drop=True)
        metadata['gpa_median'] = float(df.zfygpa.median())
        y = df.zfygpa.gt(metadata['gpa_median']).astype(int)
        s = pd.DataFrame({'race':df.race1})
        X = df[['lsat','ugpa','gender']].copy()
        metadata['race_population'] = 'Black and White students only, following AIF360 filtering'
        metadata['raw_source'] = 'https://github.com/jkomiyama/fairregresion/blob/master/dataset/law/lsac.sas7bdat'
    elif name == 'heart_disease':
        path = download('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', DATA/'revision/processed.cleveland.data')
        df = pd.read_csv(path, names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num'], na_values='?')
        y = df.pop('num').gt(0).astype(int); s = df[['sex']].copy(); X = df.drop(columns='sex')
        for c in ['cp','fbs','restecg','exang','slope','thal']: X[c] = X[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
    elif name == 'bank_marketing':
        path = download('https://archive.ics.uci.edu/static/public/222/bank+marketing.zip', DATA/'revision/bank_marketing.zip')
        with ZipFile(path) as outer:
            with ZipFile(BytesIO(outer.read('bank.zip'))) as z: df = pd.read_csv(z.open('bank-full.csv'), sep=';')
        y = df.pop('y').eq('yes').astype(int)
        # Fixed threshold avoids choosing a sensitive encoding from outer test data.
        s = pd.DataFrame({'age':df.age.ge(40).astype(int)}); X = df.drop(columns=['age','duration'])
        metadata['duration_removed'] = 'Call duration unavailable before contacting the client.'
    elif name == 'adult':
        obj = fetch_openml(data_id=1590, as_frame=True, data_home=str(DATA/'openml'), parser='auto')
        df = obj.frame.copy(); y = df.pop('class').astype(str).isin(['>50K','>50K.']).astype(int)
        s = df[['sex','race']].copy(); X = df.drop(columns=s.columns)
    elif name == 'folktables_income':
        path = DATA/'folktables/2018/1-Year/psam_p06.csv'
        cols = ['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P','PINCP','PWGTP']
        df = pd.read_csv(path, usecols=cols)
        df = df.loc[df.AGEP.gt(16) & df.PINCP.gt(100) & df.WKHP.gt(0) & df.PWGTP.ge(1)].copy()
        y = df.PINCP.gt(50000).astype(int); s = df[['SEX','RAC1P']].copy()
        X = df.drop(columns=['SEX','RAC1P','PINCP','PWGTP'])
        for c in ['COW','SCHL','MAR','OCCP','POBP','RELP']: X[c] = X[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
        metadata['survey_weights_used'] = False
    else: raise ValueError(name)
    for c in X.select_dtypes(include=['category']).columns: X[c] = X[c].astype(object)
    X = X.replace('?',np.nan).reset_index(drop=True); y = y.reset_index(drop=True); s = s.reset_index(drop=True)
    metadata.update(n=len(y), positives=int(y.sum()), features=list(X.columns), sensitive_attributes=list(s.columns))
    metadata['processed_data_sha256'] = hashlib.sha256(pd.util.hash_pandas_object(pd.concat([X,y.rename('target'),s],axis=1),index=True).values.tobytes()).hexdigest()
    if 'path' in locals(): metadata['raw_file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return X,y,s,metadata

def preprocessor(X):
    numeric = X.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in X if c not in numeric]
    return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)])
