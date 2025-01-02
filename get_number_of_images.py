import glob
import json
image_folder = '/home/erezlid/CapDec/data/images_coco/val2014/*.jpg'

images_paths = glob.glob(image_folder)
images_id = []

for path in images_paths:
    filename = path.split('/')[-1]
    filename_without_end = filename.rstrip(".jpg")
    image_id = filename_without_end.split('_')[-1]
    image_id = image_id.replace('0','')
    images_id.append(image_id)

json_file = '/home/erezlid/CapDec/post_processed_karpthy_coco/val.json'

with open(json_file, 'r') as f:
    data = json.load(f)

data_images_id = []

for image in data:
    id = image['image_id']
    data_images_id.append(str(id))

counter = 0

for image_id in images_id:
    if image_id in data_images_id:
        counter += 1

print('Number of images found: ' + str(counter))

