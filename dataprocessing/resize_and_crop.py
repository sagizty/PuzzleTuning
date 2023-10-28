from PIL import Image
import os


def resize_and_crop(source_folder, target_folder, width, height,endswith='.jpg'):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith(endswith):  # or filename.endswith(".jpg"): if some images are .jpg
            image_path = os.path.join(source_folder, filename)
            image = Image.open(image_path)

            # Crop the largest centered square
            w, h = image.size
            min_dim = min(w, h)
            left = (w - min_dim) / 2
            top = (h - min_dim) / 2
            right = (w + min_dim) / 2
            bottom = (h + min_dim) / 2
            image_cropped = image.crop((left, top, right, bottom))

            # Resize the cropped image
            image_resized = image_cropped.resize((width, height))
            target_path = os.path.join(target_folder, filename)
            image_resized.save(target_path)


source_directory = './CAM16'  # Replace this with the path to your folder with original images
target_directory = './CAM16_new'  # Replace this with the path where you want to save resized images

resize_and_crop(source_directory, target_directory, width=224, height=224, endswith='.jpg')
