import torch
import clip
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



def get_label(ID):
    # opening the file in read mode
    my_file = open('/data2/lzq/Object_Detection/dataset/train/labels/'+ID+'.txt', "r")
    
    # reading the file
    data = my_file.read()
    
    # replacing end of line('/n') with ' ' and
    # splitting the text it further when '.' is seen.
    # print(data)
    data_into_list = data.replace('\n', ' ').split(" ")
    # print(data_into_list)
    return data_into_list[0]

total_image_features=[]
total_labels = []
img_names=[]
for image_i in os.listdir('/data2/lzq/Object_Detection/dataset/train/images/'):
    ID_i = image_i[:-4]
    # print(ID_i)
    img_i='/data2/lzq/Object_Detection/dataset/train/images/'+image_i
    image = preprocess(Image.open(img_i)).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    image_features=image_features.squeeze().detach().cpu().numpy()
    total_image_features.append(image_features.reshape((1,len(image_features))))
    label_i = get_label(ID_i)
    # print(label_i)
    total_labels.append(label_i)
    img_names.append(ID_i)
    


output_features=np.concatenate(total_image_features,axis=0)

print(np.shape(output_features))
print(len(total_labels))

with open('/data/cheryl/fuwai/card/features.pkl', 'wb') as f:
    pickle.dump(output_features, f)

with open('/data/cheryl/fuwai/card/labels.pkl', 'wb') as f:
    pickle.dump(np.array(total_labels), f)

        