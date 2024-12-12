import mlflow
import mlflow 
from castle.algorithms import PC  
from . import utils
from . import model_scripts as scripts


import pandas as pd
import numpy as np
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import logging

# Configure logging

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s: %(message)s',  # Custom format without file paths
    datefmt='%Y-%m-%d %H:%M:%S', force=True  # Date format for timestamps
)


class CausalOpsEngine:

    def __init__(self,
                 mode: str,
                 algorithm: str = None,
                 max_features: int = 10,
                 algorithm_params: dict = None,
                 visualization_style: str = "network",
                 logging_target: str = "mlflow_server"):
        """
        Initialize the CausalOpsEngine.

        :param mode: Mode of operation ('create' or 'infer').
        :param algorithm: Algorithm to be used for causal discovery (mandatory in 'create' mode).
        :param max_features: Maximum number of features in the causal graph.
        :param algorithm_params: Parameters for the algorithm.
        :param visualization_style: Style for graph visualization.
        :param logging_target: Target for logging the causal model ('mlflow_server').
        """
        self.mode = mode
        self.visualization_style = visualization_style

        if self.mode == "create":
            if algorithm is None:
                raise ValueError("Algorithm is a mandatory parameter in 'create' mode.")
            if algorithm not in utils.supported_algorithms:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            self.logging_target = logging_target
            self.algorithm = algorithm
            self.model_script_path = f"{algorithm}_causal_model.py"
            self.registered_model_name = f"{algorithm}_model"
            self.pc_args = algorithm_params if algorithm_params else {}
            self.max_features = max_features
            self.model_metadata = {
                "algorithm": self.algorithm,
                "params": self.pc_args,
                "max_features": self.max_features
            }

        logging.info(f"CausalOpsEngine initialized in '{self.mode}' mode.")

    def discover_causal_graph(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Discover the causal graph from the input data.

        :param data: Input dataset as a Pandas DataFrame.
        :return: Predicted causal matrix as a Pandas DataFrame.
        """
        if self.mode != "create":
            raise ValueError("This method is only available in 'create' mode.")

        if self.algorithm not in utils.supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        logging.info(f"Discovering causal graph using '{self.algorithm}' algorithm.")
        if self.algorithm == "PC":
            pc = PC(**self.pc_args)
            pc.learn(data)
            cols = list(data.columns)
            predicted_causal_matrix_df = pd.DataFrame(pc.causal_matrix.tolist(),
                                                      columns=cols,
                                                      index=cols)
            logging.info("Causal graph discovery complete.")
            return predicted_causal_matrix_df

        raise ValueError(f"Algorithm implementation not finished: {self.algorithm}")

    def log_causal_model(self, mlflow_experiment_name: str = None) -> None:
        """
        Log the causal model and metadata to the specified logging target.

        :param mlflow_experiment_name: Name of the MLflow experiment.
        """
        if self.mode != "create":
            raise ValueError("This method is only available in 'create' mode.")

        if self.logging_target not in utils.supported_logging_target:
            raise ValueError(f"Unsupported logging target: {self.logging_target}")

        logging.info(f"Logging causal model to '{self.logging_target}'.")
        (self.input_example, self.signature) = self._infer_signature()
        args_str = self._get_model_instantiation_string()
        self._create_temp_model_file(self.algorithm, args_str)
        self.model_info = self._driver_code(mlflow_experiment_name)
        logging.info(f"Causal model logged successfully. Info: {self.model_info}")
        self._delete_temp_model_file()

        


    def _infer_signature(self):
        """
        Infer the input-output signature for the causal model.

        :return: A tuple containing:
            - `input_example`: A Pandas DataFrame example with placeholder input values.
            - `signature`: The inferred MLflow model signature.
        """
        logging.info("Inferring model signature with max_features=%d.", self.max_features)
        
        # Generate column names for the input example
        col_names = [f"x{i}" for i in range(self.max_features)]
        
        # Create an input example with placeholder values
        input_example = pd.DataFrame(np.nan, columns=col_names, index=range(5))
        
        # Generate a random output example
        output_example = np.random.choice(
            [0.0, 1.0], size=(self.max_features, self.max_features)
        )
        
        # Infer the signature using MLflow
        signature = mlflow.models.infer_signature(input_example, output_example)
        
        return (input_example, signature)


    def plot_causal_graph(
        self,
        causal_matrix_dataframe: pd.DataFrame,
        figsize: tuple = (5, 5),
        node_color: str = "tab:blue",
        edge_color: str = "yellow",
        node_size: int = 1000,
        edge_width: int = 3,
        font_size: int = 12,
        font_color: str = "white",
        layout: str = "random",
        layout_seed: int = 123
    ) -> None:
        """
        Plot the causal graph based on the provided causal matrix.

        :param causal_matrix_dataframe: DataFrame representing the causal matrix.
        :param figsize: Tuple specifying the figure size (width, height).
        :param node_color: Color of the nodes in the graph.
        :param edge_color: Color of the edges in the graph.
        :param node_size: Size of the nodes in the graph.
        :param edge_width: Width of the edges in the graph.
        :param font_size: Font size for node labels.
        :param font_color: Color of the node labels.
        :param layout: Layout algorithm for positioning the nodes in the graph.
        :param layout_seed: Seed for the layout algorithm's randomness.
        :raises ValueError: If the specified layout is not available in `utils.graph_layout_functions`.
        """
        
        # Create the plot
        fig, _ = plt.subplots(figsize=figsize)
        fig.patch.set_alpha(0)  # Set figure background to transparent

        # Extract column names and causal matrix
        column_names = list(causal_matrix_dataframe.columns)
        causal_matrix = causal_matrix_dataframe.values
        g = nx.DiGraph(causal_matrix)

        # Validate the layout
        if layout not in utils.graph_layout_functions:
            available_layouts = list(utils.graph_layout_functions.keys())
            logging.error("Invalid layout '%s'. Available options: %s", layout, available_layouts)
            raise ValueError(f"Invalid layout '{layout}'. Available options are: {available_layouts}")

        # Generate positions for the nodes using the specified layout
        pos = utils.graph_layout_functions[layout](g, seed=layout_seed)

        # Draw the graph
        nx.draw(
            G=g,
            pos=pos,
            node_color=node_color,
            edge_color=edge_color,
            node_size=node_size,
            width=edge_width,
        )

        # Draw labels
        labels = {i: column_names[i] for i in range(len(column_names))}
        nx.draw_networkx_labels(
            g, pos, labels=labels, font_size=font_size, font_color=font_color
        )

        # Print hint for layout adjustment
        print(f"Hint: for better visibility, adjust layout parameters (e.g., layout_seed), supported layouts: {list(utils.graph_layout_functions.keys())}")


    def load_causal_model(self, logged_model: str) -> None:
        """
        Load a pre-trained causal model and its metadata.

        :param logged_model: Path to the logged model in MLflow.
        """
        if self.mode != "infer":
            raise ValueError("This method is only available in 'infer' mode.")

        logging.info(f"Loading causal model from '{logged_model}'.")
        self.loaded_model = mlflow.pyfunc.load_model(logged_model)

        artifact_path = "model_metadata.json"
        try:
            local_metadata_path = mlflow.artifacts.download_artifacts(
                artifact_path=artifact_path,
                run_id=logged_model.split('/')[1]
            )
            with open(local_metadata_path, 'r') as f:
                self.model_metadata = json.load(f)

            logging.info("Causal model loaded successfully: ", self.model_metadata)
        except Exception as e:
            raise ValueError(f"Failed to load model metadata from {artifact_path}: {e}")
    
    def _get_model_instantiation_string(self) -> str:
        """
        Generate a string representation of the algorithm's parameters.

        :return: A string of key-value pairs representing the model's arguments, 
                formatted as `key=value` for each parameter in `self.pc_args`.
        """
        
        # Generate the string representation of the arguments
        args_str = ", ".join(
            f"{key}={repr(value) if isinstance(value, str) else value}"
            for key, value in self.pc_args.items()
        )
        
        # Log the generated string
        
        # Print the string (as per the original function)
        #print(args_str)
        
        return args_str
    


    def _create_temp_model_file(self, algorithm: str, args_str: str) -> None:
        """
        Create a temporary model file with the given algorithm and argument string.

        :param algorithm: The algorithm name (e.g., "PC").
        :param args_str: The string representation of the model arguments.
        :raises ValueError: If the algorithm is unsupported.
        :raises IOError: If there is an issue creating or writing the file.
        """

        # Check if the algorithm is supported
        if algorithm == "PC":
            try:
                # Retrieve the model script content based on the algorithm
                file_content = scripts.get_pc_script(args_str)
            except Exception as e:
                logging.error("Failed to generate model script for algorithm '%s': %s", algorithm, e)
                raise ValueError(f"Error generating script for algorithm '{algorithm}': {e}") from e
        else:
            raise ValueError(f"Unsupported model: {algorithm}")

        # Write the script to the temporary file
        file_name = self.model_script_path
        try:
            with open(file_name, "w") as file:
                file.write(file_content)
            logging.info("Temporary model file '%s' created successfully.", file_name)
        except IOError as e:
            logging.error("IOError while creating temporary model file '%s': %s", file_name, e)
            raise IOError(f"Error creating or writing to temporary file '{file_name}': {e}") from e
        except Exception as e:
            logging.error("Unexpected error while creating file '%s': %s", file_name, e)
            raise RuntimeError(f"Unexpected error creating file '{file_name}': {e}") from e


    def _driver_code(self, mlflow_experiment_name: str = None) :
        """
        Log the causal model to MLflow and set the experiment name.

        :param mlflow_experiment_name: Name of the MLflow experiment. If not provided, 
                                    defaults to 'causal_matrix_{algorithm}'.
        :return: Information about the logged MLflow model as a ModelInfo object.
        """
        if mlflow_experiment_name is None:
            mlflow_experiment_name = f"causal_matrix_{self.algorithm}"
            logging.info("No experiment name provided. Defaulting to '%s'.", mlflow_experiment_name)


        else:
            logging.info("Experiment name provided '%s'.", mlflow_experiment_name)

        mlflow.set_experiment(mlflow_experiment_name)

        try:
            with mlflow.start_run():

                # Log the model
                logging.info("Logging model artifacts.")
                model_info = mlflow.pyfunc.log_model(
                    artifact_path="model",  
                    python_model=self.model_script_path,  
                    signature=self.signature,
                    input_example=self.input_example,  
                    registered_model_name=self.registered_model_name  
                )
                # Log metadata
                mlflow.log_dict(self.model_metadata, "model_metadata.json")

            return model_info
        except Exception as e:
            logging.error("Error during MLflow logging: %s", e)
            raise RuntimeError(f"Failed to log model to MLflow: {e}") from e



    def _delete_temp_model_file(self) -> None:
        """
        Delete the temporary model file created during the process.

        :raises FileNotFoundError: If the file does not exist.
        """
        file_name = self.model_script_path

        if os.path.exists(file_name):
            try:
                os.remove(file_name)
                logging.info("Temporary file '%s' deleted successfully.", file_name)
            except Exception as e:
                logging.error("Failed to delete temporary file '%s': %s", file_name, e)
                raise RuntimeError(f"Error while deleting temporary file '{file_name}': {e}") from e
        else:
            logging.warning("Temporary file '%s' does not exist, cannot delete.", file_name)

    def _map_schema(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Map the schema of the input data to standardized column names.

        :param data: Input data as a Pandas DataFrame.
        :return: A new DataFrame with standardized column names (x0, x1, ..., xn).
        """

        # Generate new column names
        new_columns = [f"x{i}" for i in range(data.shape[1])]

        # Copy and rename the data
        renamed_data = data.copy()
        renamed_data.columns = new_columns
        return renamed_data
    

    def load_causal_model1(self, logged_model: str) -> None:
        """
        Load a causal model from the specified MLflow location.

        :param logged_model: The MLflow model URI to load the causal model from.
        :raises ValueError: If the method is called in 'create' mode.
        """
        logging.info("Attempting to load causal model from '%s'.", logged_model)

        if self.mode == "infer":
            # Load the model using MLflow
            self.loaded_model = mlflow.pyfunc.load_model(logged_model)
            logging.info("Causal model loaded successfully in 'infer' mode.")
        elif self.mode == "create":
            logging.error("Invalid mode: 'load_causal_model' is only available in 'infer' mode.")
            raise ValueError("'load_causal_model' only available in 'infer' mode.")
        
    def infer_causal_model(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Infer the causal model by predicting the causal relationships based on input data.

        :param data: Input data as a Pandas DataFrame.
        :return: A DataFrame representing the predicted causal matrix with the original schema.
        :raises ValueError: If the method is called in 'create' mode.
        """

        if self.mode == "infer":
            # Save the original schema
            self.original_schema = list(data.columns)
            logging.info("Data schema extracted: %s", self.original_schema)

            # Map schema to standard format
            mapped_data = self._map_schema(data)
            logging.info("Schema mapped for causal model inference.")

            # Predict the causal matrix
            predicted_causal_matrix = self.loaded_model.predict(pd.DataFrame(mapped_data))

            # Convert prediction to DataFrame with original schema
            predicted_causal_matrix_df = pd.DataFrame(
                predicted_causal_matrix.tolist(),
                columns=self.original_schema,
                index=self.original_schema
            )

            return predicted_causal_matrix_df
        elif self.mode == "create":
            logging.error("Invalid mode: 'infer_causal_model' is only available in 'infer' mode.")
            raise ValueError("'infer_causal_model' only available in 'infer' mode.")
