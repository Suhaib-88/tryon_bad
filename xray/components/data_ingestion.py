from xray.constants.training_pipeline import *
from xray.entity.artifact_entity import DataIngestionArtifact
from xray.entity.config_entity import DataIngestionConfig
from xray.cloud_storage.s3_operation import S3Operation
from xray.exception import XRayException
from xray.logger import logging
import sys


class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config=data_ingestion_config
        self.s3_operations=S3Operation()

    def get_data_from_s3(self):
        try:
            self.s3_operations.sync_folder_from_s3()
            logging.info("Exited the get_data_from_s3 method of Data ingestion class")

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_data_ingestion(self):
        try:
            self.get_data_from_s3()
            data_ingestion_artifact:DataIngestionArtifact= DataIngestionArtifact(training_file_path=self.data_ingestion_config.train_data_path,testing_file_path=self.data_ingestion_config.test_data_path)
            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")
            return data_ingestion_artifact
        except Exception as e:
            raise XRayException(e, sys)