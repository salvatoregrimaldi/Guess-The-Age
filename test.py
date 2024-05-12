import os, sys, cv2, csv, argparse, random
import numpy as np

from tensorflow import keras

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--data", type=str, default='foo_test.csv', help="Dataset labels")
    parser.add_argument("--images", type=str, default='foo_test/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results.csv', help="CSV file of the results")
    args = parser.parse_args()
    return args

def expectation_regression(y):
  k = np.linspace(np.ones(y.shape[0]), np.ones(y.shape[0])*100, 100, axis=1)
  y_exp = np.sum(y * k, axis=1)
  return y_exp

args = init_parameter()

#Load model and weights
model = keras.models.load_model('A3MoreDenseFold0_class')

# Reading CSV test file
with open(args.data, mode='r') as csv_file:
    gt = csv.reader(csv_file, delimiter=',')
    gt_num = 0
    gt_dict = {}
    for row in gt:
        gt_dict.update({row[0]: int(round(float(row[1])))})
        gt_num += 1
print(gt_num)

# Opening CSV results file
with open(args.results, 'w+', newline='') as res_file:
    writer = csv.writer(res_file)
    # Processing all the images
    for image in gt_dict.keys():
        img = keras.utils.img_to_array(keras.utils.load_img(args.images+image, target_size = (224, 224), interpolation = "bilinear"))
        if img.size == 0:
            print("Error")
        # Make the prediction on the single image
        img = np.expand_dims(img, axis=0)
        age = model(img, training=False)
        age = expectation_regression(age)[0]
        # Writing a row in the CSV file
        writer.writerow([image, age])