import os
import yaml
import pandas as pd

def load_dataset(owd: str, dataset_name: str, dataset_info: dict, show_info: bool = False) -> (pd.DataFrame, pd.Series):

    # Load file
    file_path = '{}/dataset/{}/{}'.format(owd, dataset_name, dataset_info['filename'])
    X = pd.read_csv(file_path, sep=dataset_info['csv_sep'], header=dataset_info['header'])

    # Update columns
    feature_cols = []
    if dataset_info['target_col_index'] == -1:
        target_col_index = len(X.columns) - 1 
    else:
        target_col_index = dataset_info['target_col_index']

    # Select feature columns
    new_col_index = 0
    mapping_dict = {} 
    for i, c in enumerate(X.columns):

        if show_info:
            print()
            print(f'{i}: {c}')
            print('- dtype', X[c].dtype)
            print('- first 10 values', X[c].values[:10])
            if i in dataset_info['remove_col_index']:
                print(f'col {c} removed')

        if i not in dataset_info['remove_col_index']:
            if i == target_col_index:
                new_col_name = 'target'
            else:
                new_col_name = f'x_{new_col_index}'
                new_col_index += 1

            feature_cols.append(new_col_name)
            mapping_dict[new_col_name] = c 

    X = X[mapping_dict.values()]
    X.columns = mapping_dict.keys()

    if show_info:
        print(mapping_dict)
        if dataset_info['task'] == 'class':
            print(X['target'].unique())

    y = X['target']
    X.drop('target', axis=1, inplace=True)

    return X, y



def main():
    owd = os.getcwd() 
    dataset_name = 'california-housing-prices'

    with open(f'{owd}/conf/dataset_conf.yaml', 'r') as yml:
        dataset_conf = yaml.safe_load(yml)

    dataset_info = dataset_conf[dataset_name]
    X, y = load_dataset(owd, dataset_name, dataset_info, show_info=True)

    return

if __name__=='__main__':
    main()
