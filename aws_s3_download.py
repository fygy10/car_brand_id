import boto3
import os
from PIL import Image


class S3:
    

    #DOWNLOAD TEST INTANCE
    def example_image(self, bucket_name, object_key, file_path):

        s3 = boto3.client('s3')
        s3.download_file(bucket_name, object_key, file_path)

        img = Image.open(file_path)
        img.show()



    #PULL TRAINING DATA
    @staticmethod
    def download_train_selections(bucket_name, training_file_path, local_directory):
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')

        downloaded_files = 0
        for page in paginator.paginate(Bucket=bucket_name, Prefix=training_file_path):
            for obj in page.get('Contents', []):
                if downloaded_files == 4000:
                    return
                filename = os.path.basename(obj['Key'])
                local_path = os.path.join(local_directory, filename)
                s3.download_file(bucket_name, obj['Key'], local_path)
                downloaded_files += 1


    #PULL TEST DATA
    @staticmethod
    def download_test_selections(bucket_name, testing_file_path, local_directory):
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')

        downloaded_files = 0
        for page in paginator.paginate(Bucket=bucket_name, Prefix=testing_file_path):
            for obj in page.get('Contents', []):
                if downloaded_files == 1000:
                    return
                filename = os.path.basename(obj['Key'])
                local_path = os.path.join(local_directory, filename)
                s3.download_file(bucket_name, obj['Key'], local_path)
                downloaded_files += 1




#Global
download = S3()
bucket_name  = 'car-id-database'


#OLDER CODE FROM TESTING ITERATIONS
# local_directory = ''
# training_object_key = 'archive/cars_train/cars_train'
# download.download_train_selections(bucket_name, training_file_path=training_object_key, local_directory=local_directory)


# local_test_directory = '
# testing_object_key = 'archive/cars_test/cars_test'
# download.download_test_selections(bucket_name, testing_file_path=testing_object_key, local_directory=local_test_directory)



#Example
# example_object_key = 'archive/cars_test/cars_test/00001.jpg'
# example_file_path = os.path.join('/Users/tim/Desktop', '00001.jpg')
# download.example_image(bucket_name, example_object_key, example_file_path)


# #Training
# training_object_key = 'archive/cars_train/cars_train'
# training_file_path = os.path.join('/Users/tim/Documents/Education/Harvard/Artificial_Intelligence/Final_Project/train_images', training_object_key)
# download.training_data(bucket_name, training_object_key, training_file_path)


# #Testing
# testing_object_key = 'archive/cars_test/cars_test'
# testing_file_path = os.path.join('/Users/tim/Documents/Education/Harvard/Artificial_Intelligence/Final_Project/test_images', testing_object_key)
# download.testing_data(bucket_name, testing_object_key, testing_file_path)