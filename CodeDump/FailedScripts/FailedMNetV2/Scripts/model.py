import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MNetV2(nn.Module): #MultiOutputModel
    def __init__(self, l_class): #n_color_classes, n_gender_classes, n_article_classes
        super().__init__()
        self.base_model = models.mobilenet_v2().features  # take the model without classifier
        last_channel = models.mobilenet_v2().last_channel  # size of the layer before classifier

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.letter = nn.Sequential( #color
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=l_class) #n_color_classes
        )
        # self.gender = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=last_channel, out_features=n_gender_classes)
        # )
        # self.article = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=last_channel, out_features=n_article_classes)
        # )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return {
            'letter': self.letter(x) #,
            # 'gender': self.gender(x),
            # 'article': self.article(x)
        }

    def get_loss(self, net_output, ground_truth):
        letter_loss = F.cross_entropy(net_output['letter'], ground_truth['letter_labels']) #color
        # gender_loss = F.cross_entropy(net_output['gender'], ground_truth['gender_labels'])
        # article_loss = F.cross_entropy(net_output['article'], ground_truth['article_labels'])
        loss = letter_loss #+ gender_loss + article_loss
        return loss, {'letter': letter_loss} #, {'color': color_loss, 'gender': gender_loss, 'article': article_loss}
