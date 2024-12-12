
def get_pc_script(args_str):
    pc_script = f"""
import mlflow 
import pandas as pd 
from mlflow.pyfunc import PythonModel
from mlflow.models import set_model
from castle.algorithms import PC

class PCCausalModel(PythonModel):

    def predict(self, context, model_input):
        pc=PC({args_str})
        pc.learn(model_input)  
        return pc.causal_matrix 
set_model(model=PCCausalModel())
"""
    return(pc_script)