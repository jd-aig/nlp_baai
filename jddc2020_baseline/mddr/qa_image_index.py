import argparse
import PIL
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from indexers.faiss_indexers import *
from tqdm import tqdm

# temporarily use resent18 image statistics
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'eval': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ImageDataset(Dataset):
    def __init__(self, data, data_type='eval'):
        self.data = data
        self.data_type = data_type
        self.len = len(data)

    def __getitem__(self, index):
        image = self.data[index]
        image = self.image_transform(image, self.data_type)
        return image

    def __len__(self):
        return self.len

    @staticmethod
    def image_transform(image, data_type):
        img = torch.zeros(3, 224, 224)
        try:
            img_tmp = PIL.Image.open(image)
            img = data_transforms[data_type](img_tmp)
        except Exception as err:
            print(err)

        return img


def get_loader(data, batch_size=256, data_type='eval'):

    dataset = ImageDataset(data=data, data_type=data_type)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False)

    return data_loader


def get_imges_embedding(img_data_loader):

    model_ft = torchvision.models.resnet18(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])
    feature_extractor.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_extractor.to(device)

    flag_s = 0
    for batch_i, (images) in enumerate(tqdm(img_data_loader, ncols=80)):

        images = images.to(device)
        model_out = feature_extractor(images)
        model_data = model_out.cpu().data.numpy().squeeze()
        if not flag_s:
            data_s = model_data
            flag_s = 1
        else:
            data_s = np.vstack((data_s, model_data))

    return data_s


def create_image_embedding_idx(qa_file, dir, index_file):
    img_list = list()

    with open(qa_file) as f:
        lines = f.readlines()

    for line in lines:
        word = line.strip().split('\t')
        ans = word[0]
        ques = word[1]
        img_list.append(dir+ans)

    img_data_loader = get_loader(img_list)

    data_s = get_imges_embedding(img_data_loader)

    index = DenseFlatIndexer(512)

    buffer = []
    for idx in range(0, data_s.shape[0]):
        buffer.append((idx, data_s[idx]))
        if 0 < 50000 == len(buffer):
            index.index_data(buffer)
            buffer = []
    index.index_data(buffer)

    index.serialize(index_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tool to create image embedding index")

    parser.add_argument('-f', '--img_qa_file', default='img_QA_dbs.txt')
    parser.add_argument('-d', '--img_dir', default='./data/images/train/')
    parser.add_argument('-i', '--img_index_file', default='jddc_img')

    args = parser.parse_args()

    create_image_embedding_idx(args.img_qa_file, args.img_dir, args.img_index_file)