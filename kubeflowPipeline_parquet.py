import kfp

from kfp import dsl
import kfp.compiler
from kfp.dsl import OutputPath, InputPath
# from kfp.components import func_to_container_op

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'numpy==2.0.0', 'requests==2.32.3', 'pyarrow==15.0.0']
)
def load_data(load_data_output: OutputPath()):
    import pandas as pd
    import numpy as np
    import io
    import requests
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    url = "https://raw.githubusercontent.com/daniel88516/diabetes-data/main/10k.csv"
    s = requests.get(url).content
    df_data = pd.read_csv(io.StringIO(s.decode('utf-8')))
    df_data.drop(df_data[df_data['diabetes'] == 'No Info'].index, inplace=True)
    df_data = df_data[['gender','age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']]
    df_data.dropna(thresh=4, inplace=True)
    
    gender_map = {'Male':0 , 'Female':1  , 'Other':2}
    df_data['gender'] = df_data['gender'].map(gender_map)
    df_data = df_data[df_data['gender'] != 2]
    df_data['age'] = pd.to_numeric(df_data['age'].replace('No Info', np.nan), errors='coerce')
    df_data['bmi'] = pd.to_numeric(df_data['bmi'].replace('No Info', np.nan), errors='coerce')
    df_data['HbA1c_level'] = pd.to_numeric(df_data['HbA1c_level'].replace('No Info', np.nan), errors='coerce')
    df_data['blood_glucose_level'] = pd.to_numeric(df_data['blood_glucose_level'].replace('No Info', np.nan), errors='coerce')
    
    df_data = df_data.fillna(df_data.mean())
    
    # 轉換為 Parquet 格式
    table = pa.Table.from_pandas(df_data)
    pq.write_table(table, load_data_output)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'pyarrow==15.0.0']
)
def prepare_data(data: InputPath(), x_train_output: OutputPath(), x_test_output: OutputPath(), y_train_output: OutputPath(), y_test_output: OutputPath()):
    import pandas as pd
    import pyarrow.parquet as pq
    from sklearn.model_selection import train_test_split
    
    df_data = pq.read_table(data).to_pandas()
    x = df_data.drop(labels=['diabetes'], axis=1)
    y = df_data[['diabetes']]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    pq.write_table(pa.Table.from_pandas(x_train), x_train_output)
    pq.write_table(pa.Table.from_pandas(x_test), x_test_output)
    pq.write_table(pa.Table.from_pandas(y_train), y_train_output)
    pq.write_table(pa.Table.from_pandas(y_test), y_test_output)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'joblib==1.4.2', 'pyarrow==15.0.0']
)
def train_model(x_train: InputPath(), y_train: InputPath(), train_model_output: OutputPath()):
    import pandas as pd
    import pyarrow.parquet as pq
    from sklearn.linear_model import LogisticRegression
    import joblib
    
    x_train = pq.read_table(x_train).to_pandas()
    y_train = pq.read_table(y_train).to_pandas()
    
    model = LogisticRegression(random_state=0, max_iter=100) # 100 times for test p.s. it is 10000 times in beginning
    model.fit(x_train, y_train.values.ravel())
    
    joblib.dump(model, train_model_output)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'joblib==1.4.2', 'pyarrow==15.0.0']
)
def evaluate_model(model_path: InputPath(), x_test: InputPath(), y_test: InputPath()) -> str:
    import pandas as pd
    import pyarrow.parquet as pq
    import joblib
    
    model = joblib.load(filename=model_path)
    x_test_df = pq.read_table(x_test).to_pandas()
    y_test_df = pq.read_table(y_test).to_pandas()
    
    accuracy = model.score(x_test_df, y_test_df)
    
    return f'Test accuracy: {accuracy}'

@dsl.pipeline(
    name='Diabetes Prediction Pipeline',
    description='Using kubeflow pipeline to train and evaluate a diabetes prediction model'
)
def diabetes_prediction_pipeline() -> str:
    load_data_task = load_data()
    prepare_data_task = prepare_data(data=load_data_task.outputs['load_data_output'])
    
    train_model_task = train_model(
        x_train = prepare_data_task.outputs['x_train_output'], 
        y_train = prepare_data_task.outputs['y_train_output']
    )
    
    evaluate_model_task = evaluate_model(
        model_path = train_model_task.outputs['train_model_output'], 
        x_test = prepare_data_task.outputs['x_test_output'], 
        y_test = prepare_data_task.outputs['y_test_output']
    )
    
    return evaluate_model_task.output

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(diabetes_prediction_pipeline, 'diabetes_prediction_pipeline_parquet.yaml')