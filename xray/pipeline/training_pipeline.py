from xray.components.data_ingestion import DataIngestion
from xray.components.data_transformation import DataTransformation
from xray.components.model_training import ModelTrainer
from xray.entity.config_entity import DataIngestionConfig, DataTransformationConfig
from xray.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact,ModelTrainerArtifact
from xray.exception import XRayException
from xray.logger import logging
import os,sys

class TrainPipeline:
    def __init__(self) -> None:
        self.data_ingestion_config=DataIngestionConfig
        self.data_transformation_config = DataTransformationConfig()

    def start_data_ingestion(self):
        try:
            self.data_ingestion= DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact=self.data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from s3")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")

            return data_ingestion_artifact

        except Exception as e:
            raise XRayException(e, sys)
        
    def start_data_transformation(self, data_ingestion_artifact:DataIngestionArtifact)-> DataTransformationArtifact:
        try:
            data_transformation= DataTransformation(data_ingestion_config=self.data_ingestion_config,data_transformation_config=self.data_transformation_config)
            data_transformation_artifact=(data_transformation.initiate_data_transformation())          
            logging.info(
                "Exited the start_data_transformation method of TrainPipeline class"
            )

            return data_transformation_artifact
        except Exception as e:
            raise XRayException(e, sys)
        
    def start_model_trainer(self,  data_transformation_artifact: DataTransformationArtifact)-> ModelTrainerArtifact:
        try:
            model_trainer= ModelTrainer(data_transformation_artifact=data_transformation_artifact,model_trainer_config=self.model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()          
            logging.info(
                "Exited the start_data_transformation method of TrainPipeline class"
            )

            return data_transformation_artifact
        except Exception as e:
            raise XRayException(e, sys)
        

    def run_pipeline(self) -> None:
        logging.info("Entered the run_pipeline method of TrainPipeline class")

        try:
            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            data_transformation_artifact: DataTransformationArtifact = (self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact))
            model_trainer_artifact: ModelTrainerArtifact = (self.start_model_trainer(data_transformation_artifact=data_transformation_artifact))
           
           
            logging.info("Exited the run_pipeline method of TrainPipeline class")

        except Exception as e:
            raise XRayException(e, sys)