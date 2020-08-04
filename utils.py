import os
import requests
import imageio
import cv2


def get_image(data_dict, file_name):
    image_url = data_dict["Labeled Data"]
    res = requests.get(image_url, timeout=20)
    with open(file_name, "wb") as f:
        f.write(res.content)


def get_label(data_dict, path, label):
    index = 1
    path = path + label + "/"
    if os.path.exists(path) is False:
        os.makedirs(path)
    try:
        for obj in data_dict["Label"]["objects"]:
            if obj["value"] == label:
                img = requests.get(obj["instanceURI"], timeout=20)
                with open(path + str(index) + ".jpg", "wb") as f:
                    f.write(img.content)
                index += 1
    except:
        print("there is no object")