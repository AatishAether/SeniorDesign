import argparse
import os
from datetime import datetime

import torch
import torchvision.transforms as transforms
#from dataset import FashionDataset, AttributesDataset, mean, std
from main import ASLDataset, mean, std, AttributesDataset
# from model import MultiOutputModel
from model import MNetV2
from test import calculate_metrics, validate, visualize_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--attributes_file', type=str, default='./train.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    start_epoch = 1
    N_epochs = 50
    batch_size = 25
    num_workers = 8  # number of processes to handle dataset loading
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

    # specify image transforms for augmentation during training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2),
                                shear=None), # , resample=False, fillcolor=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # train_dataset = FashionDataset('./train.csv', attributes, train_transform)
    train_dataset = ASLDataset('./train.csv', attributes, train_transform) #, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # val_dataset = FashionDataset('./val.csv', attributes, val_transform)
    val_dataset = ASLDataset('./test.csv', attributes, val_transform)  #val.csv
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(device)

    model = MNetV2(l_class=attributes.num_letters).to(device) #MultiOutputModel ,
                             # n_gender_classes=attributes.num_genders,
                             # n_article_classes=attributes.num_articles)\
    print(torch.cuda.is_available())
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters())

    logdir = os.path.join('./logs/', get_cur_time())
    savedir = os.path.join('./checkpoints/', get_cur_time())
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    n_train_samples = len(train_dataloader)

    # Uncomment rows below to see example images with ground truth labels in val dataset and all the labels:
    # visualize_grid(model, val_dataloader, attributes, device, show_cn_matrices=False, show_images=True,
    #                checkpoint=None, show_gt=True)
    # print("\nAll gender labels:\n", attributes.gender_labels)
    # print("\nAll color labels:\n", attributes.color_labels)
    # print("\nAll article labels:\n", attributes.article_labels)

    print("Starting training ...")

    for epoch in range(start_epoch, N_epochs + 1):
        total_loss = 0
        accuracy_letter = 0 #color
        # accuracy_gender = 0
        # accuracy_article = 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
            batch_accuracy_letter = calculate_metrics(output, target_labels) # \ color, batch_accuracy_gender, batch_accuracy_article


            accuracy_letter += batch_accuracy_letter #batch_accuracy_color color
            # accuracy_gender += batch_accuracy_gender
            # accuracy_article += batch_accuracy_article

            loss_train.backward()
            optimizer.step()

        print("epoch {:4d}, loss: {:.4f}, letter: {:.4f}".format(epoch, total_loss, accuracy_letter))

        # color: {:.4
        # f}, gender: {:.4
        # f}, article: {:.4
        # f}".format(
        # epoch,
        # total_loss / n_train_samples,
        # accuracy_color / n_train_samples,
        # accuracy_gender / n_train_samples,
        # accuracy_article / n_train_samples))

        logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)

        if epoch % 5 == 0:
            validate(model, val_dataloader, logger, epoch, device)

        if epoch % 25 == 0:
            checkpoint_save(model, savedir, epoch)
