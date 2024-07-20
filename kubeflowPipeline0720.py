import kfp

from kfp import dsl
import kfp.compiler
from kfp.dsl import OutputPath, InputPath
# from kfp.components import func_to_container_op

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2'])
def load_data(load_data_output: OutputPath(str)):
    import pandas as pd
    
    url = "https://raw.githubusercontent.com/daniel88516/diabetes-data/main/10k.csv"
    df_data = pd.read_csv(url)
    
    df_data = df_data.drop(df_data[df_data['diabetes'] == 'No Info'].index)
    df_data = df_data[['gender','age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']]
    df_data = df_data.dropna(thresh=4)
    
    gender_map = {'Male': 0 , 'Female': 1  , 'Other': 2}
    df_data['gender'] = df_data['gender'].map(gender_map)
    df_data = df_data[df_data['gender'] != 2]
    df_data['age'] = df_data['age'].replace('No Info', df_data['age'].mean())
    df_data['bmi'] = df_data['bmi'].replace('No Info', df_data['bmi'].mean())
    df_data['HbA1c_level'] = df_data['HbA1c_level'].replace('No Info', df_data['HbA1c_level'].mean())
    df_data['blood_glucose_level'] = df_data['blood_glucose_level'].replace('No Info', df_data['blood_glucose_level'].mean())

    df_data.to_csv(load_data_output, index=False)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1']
)
def prepare_data(
    data: InputPath(str), 
    x_train_output: OutputPath(str), x_test_output: OutputPath(str),
    y_train_output: OutputPath(str), y_test_output: OutputPath(str)
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df_data = pd.read_csv(data)
    x = df_data.drop(labels=['diabetes'], axis=1)
    y = df_data[['diabetes']]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    x_train_df = pd.DataFrame(x_train)
    x_test_df = pd.DataFrame(x_test)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)

    x_train_df.to_csv(x_train_output, index=False)
    x_test_df.to_csv(x_test_output, index=False)
    y_train_df.to_csv(y_train_output, index=False)
    y_test_df.to_csv(y_test_output, index=False)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1', 'joblib==1.4.2']
)
def train_model(x_train: InputPath(str), y_train: InputPath(str), train_model_output: OutputPath(str)):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    import joblib
    
    x_train = pd.read_csv(x_train)
    y_train = pd.read_csv(y_train)
    
    model = LogisticRegression(random_state=0, max_iter=100) # 100 times for test p.s. it is 10000 times in beginning
    model.fit(x_train, y_train)
    
    #model_path = './diabete_prediction_model.pkl'
    joblib.dump(model, train_model_output)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'joblib==1.4.2']
)
def evaluate_model(model_path: InputPath(str), x_test: InputPath(str), y_test: InputPath(str)) -> str:
    import pandas as pd
    import joblib

    model = joblib.load(filename=model_path)

    x_test_df = pd.read_csv(x_test)
    y_test_df = pd.read_csv(y_test)
    
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
    kfp.compiler.Compiler().compile(diabetes_prediction_pipeline, 'diabetes_prediction_pipeline.yaml')