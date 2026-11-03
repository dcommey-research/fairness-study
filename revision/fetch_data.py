"""Fetch the declared raw datasets without fitting any preprocessing."""
from revision.data import DATA,DATASETS,load
from folktables import ACSDataSource

def main():
    path=DATA/'folktables/2018/1-Year/psam_p06.csv'
    if not path.exists():
        ACSDataSource(survey_year='2018',horizon='1-Year',survey='person',root_dir=str(DATA/'folktables')).get_data(states=['CA'],download=True)
    for name in DATASETS:
        _,_,_,metadata=load(name)
        print(name,metadata['n'],metadata['processed_data_sha256'])

if __name__=='__main__':main()
