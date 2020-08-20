import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from utils import PAD_ID, UNK_ID, SOS_ID, EOS_ID
import PIL
import torch
from torchvision import datasets, models, transforms


# temporarily use resent18 image statistics
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class DialogDataset(Dataset):
    def __init__(self, sentences, images, conv_img_length, conversation_length, sentence_length, vocab, data=None, data_type='val'):

        # [total_data_size, max_conversation_length, max_sentence_length]
        # tokenized raw text of sentences
        self.sentences = sentences
        self.vocab = vocab

        # conversation length of each batch
        # [total_data_size]
        self.conversation_length = conversation_length

        self.images = images
        self.conv_img_length = conv_img_length

        # list of length of sentences
        # [total_data_size, max_conversation_length]
        self.sentence_length = sentence_length
        self.data = data
        self.len = len(sentences)
        self.data_type = data_type

    def __getitem__(self, index):
        """Return Single data sentence"""
        # [max_conversation_length, max_sentence_length]
        sentence = self.sentences[index]
        conversation_length = self.conversation_length[index]
        sentence_length = self.sentence_length[index]

        # word => word_ids
        sentence = self.sent2id(sentence)
        image = self.images[index]
        image = self.image_transform(image, self.data_type)
        image_length = self.conv_img_length[index]

        return sentence, conversation_length, sentence_length, image, image_length

    def __len__(self):
        return self.len

    def sent2id(self, sentences):
        """word => word id"""
        # [max_conversation_length, max_sentence_length]
        return [self.vocab.sent2id(sentence) for sentence in sentences]

    def image_transform(self, images, data_type):
        resp_list = list()

        for image in images:
            if image == "NULL":
                img = torch.zeros(3, 224, 224)
                resp_list.append(img)
            else:
                img = torch.zeros(3, 224, 224)
                try:
                    img_tmp = PIL.Image.open(image)
                    img = data_transforms[data_type](img_tmp)
                except:
                    # print("can't open image file: ", image)
                    None
                finally:
                    resp_list.append(img)

        return resp_list


def get_loader(sentences, images, conv_img_length, conversation_length, sentence_length, vocab, batch_size=100,
               data=None, shuffle=True, data_type='eval'):
    """Load DataLoader of given DialogDataset"""

    def collate_fn(data):
        """
        Collate list of data in to batch

        Args:
            data: list of tuple(source, target, conversation_length, source_length, target_length)
        Return:
            Batch of each feature
            - source (LongTensor): [batch_size, max_conversation_length, max_source_length]
            - target (LongTensor): [batch_size, max_conversation_length, max_source_length]
            - conversation_length (np.array): [batch_size]
            - source_length (LongTensor): [batch_size, max_conversation_length]
        """
        # Sort by conversation length (descending order) to use 'pack_padded_sequence'
        data.sort(key=lambda x: x[1], reverse=True)

        # Separate
        sentences, conversation_length, sentence_length, images, conv_img_length = zip(*data)

        # return sentences, conversation_length, sentence_length.tolist()
        return sentences, conversation_length, sentence_length, images, conv_img_length

    dataset = DialogDataset(sentences, images, conv_img_length, conversation_length,
                            sentence_length, vocab, data=data, data_type=data_type)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader
