from xray.constants.training_pipeline import *
from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader

@dataclass
class DataIngestionArtifact:
    training_file_path:str
    testing_file_path:str

@dataclass
class DataTransformationArtifact:
    transformed_train_object: DataLoader

    transformed_test_object: DataLoader

    train_transform_file_path: str

    test_transform_file_path: str

@dataclass
class ModelTrainerArtifact:
    trained_model_path: str