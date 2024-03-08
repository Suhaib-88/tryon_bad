import os,sys
from xray.exception import XRayException
import gdown

class S3Operation:
    def sync_folder_to_s3(
        self, folder: str, bucket_name: str, bucket_folder_name: str
    ) -> None:
        try:
            command: str = (
                f"aws s3 sync {folder} s3://{bucket_name}/{bucket_folder_name}/ "
            )

            os.system(command)

        except Exception as e:
            raise XRayException(e, sys)

    def sync_folder_from_s3(
        self
    ) -> None:
        try:
            # command: str = (
            #     f"aws s3 sync s3://{bucket_name}/{bucket_folder_name}/ {folder} "
            # )

            # os.system(command)
            gdown.download_folder("https://drive.google.com/drive/folders/16tweNm94MsIzmELXUq7OLO8wAIM8AxsL?usp=sharing", quiet=True)
        except Exception as e:
            raise XRayException(e, sys)