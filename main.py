import argparse
import os
import cv2
import csv
import copy
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from nets import nn
from PIL import Image
from utils import util
from utils.dataset import Datasets
from face_detection import RetinaFace
from torchvision import transforms


def train(args):
    dataset = Datasets(args, is_train=True)
    loader = DataLoader(dataset, args.batch_size, True, num_workers=8)
    model = nn.GazeNet(args)
    model.cuda()

    best = float('inf')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_patience, gamma=args.lr_decay)

    util.model_info(model)

    with open('./weights/log.csv', 'w') as log:
        logger = csv.DictWriter(log, fieldnames=['epoch', 'MAE'])
        logger.writeheader()

        for epoch in range(args.epochs):
            model.train()
            p_bar = loader
            MAE = util.AverageMeter()

            print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'MAE'))
            p_bar = tqdm.tqdm(iterable=p_bar, total=len(loader))

            for images, labels in p_bar:
                image = images["face"].cuda()
                label = labels.cuda()
                with torch.no_grad():
                    model.eval()
                    pred_gaze0 = model(torch.flip(image, (3,)))
                    pred_gaze_ = pred_gaze0.clone()
                    pred_gaze_[:, 0] = -pred_gaze0[:, 0]

                model.train()
                pred = model(image)
                gaze_error = np.mean(util.angular_error(pred.cpu().data.numpy(), label.cpu().data.numpy()))
                MAE.update(gaze_error.item(), image.size()[0])

                # L1 Gaze loss
                loss_gaze = F.l1_loss(pred, label)
                # LR Symmetry loss
                loss_sym = F.l1_loss(pred, pred_gaze_)

                loss_all = loss_gaze + loss_sym

                optimizer.zero_grad()
                loss_all.backward()
                optimizer.step()

                memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                s = ('%10s' * 2 + '%10.3g') % (f'{epoch + 1}/{args.epochs}', memory, MAE.avg)
                p_bar.set_description(s)
            scheduler.step()

            last = test(args, model)
            logger.writerow({'epoch': str(epoch + 1).zfill(3),
                             'MAE': str(f'{last:.3f}')})

            log.flush()
            is_best = last < best

            if is_best:
                best = last
            save = {'epoch': epoch, 'model': copy.deepcopy(model).half()}
            torch.save(save, f'weights/last.pt')

            if is_best:
                torch.save(save, f'weights/best.pt')
            del save

        util.strip_optimizer('weights/best.pt')
        util.strip_optimizer('weights/last.pt')
        torch.cuda.empty_cache()


def test(args, model=None):
    if model is None:
        model = torch.load('weights/best.pt', 'cuda')
        model = model['model'].float()
    model.cuda().eval()

    dataset = Datasets(args, is_train=False)
    loader = DataLoader(dataset, args.batch_size, False, num_workers=4)

    MAE = util.AverageMeter()
    with torch.no_grad():
        for images, labels in tqdm.tqdm(loader, '%10s' % 'MAE'):
            image = images["face"].cuda()

            pred = model(image)

            gaze_error_batch = np.mean(util.angular_error(pred.cpu().data.numpy(), labels.data.numpy()))
            MAE.update(gaze_error_batch.item(), image.size()[0])

    print('%10s' % f'{MAE.avg:.3f}')
    model.float()  # for training
    return MAE.avg


def meta_train(args):
    train_dataset = Datasets(args, is_train=True)
    val_dataset = Datasets(args, is_train=False)
    train_loader = DataLoader(train_dataset, args.batch_size, True, num_workers=8)
    val_loader = DataLoader(val_dataset, args.batch_size, False, num_workers=4)

    ckpt = torch.load('weights/best.pt', map_location='cuda')
    model = ckpt['model'].float()

    util.freeze_params(model)
    util.make_padding_trainable(model)

    for name, w in model.named_parameters():
        if 'prompt' in name:
            w.requires_grad = True

    model.cuda()
    util.model_info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_patience, gamma=args.lr_decay)

    # model.eval()
    prev_model = os.path.join('weights', 'tem_model.pth')
    torch.save(model.state_dict(), prev_model)
    inner_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    index = -1
    best = float('inf')
    with open('./weights/meta_log.csv', 'w') as log:
        logger = csv.DictWriter(log, fieldnames=['epoch', 'MAE'])
        logger.writeheader()
        for epoch in range(args.epochs):
            p_bar = tqdm.tqdm(train_loader, total=len(train_loader))

            for images, labels in p_bar:
                image = images["face"].cuda()
                label = labels.cuda()
                index += 1

                if index % 2 == 0:
                    dummy_input = image[0].unsqueeze(0)
                    dummy_gt = label[0].unsqueeze(0)

                    with torch.no_grad():
                        pred_gaze0 = model(torch.flip(image, (3,)))
                        pred_gaze_ = pred_gaze0.clone()
                        pred_gaze_[:, 0] = -pred_gaze0[:, 0]

                    pred_gaze = model(image)
                    loss_all = F.l1_loss(pred_gaze, pred_gaze_)

                    inner_optimizer.zero_grad()
                    loss_all.backward()
                    inner_optimizer.step()

                else:
                    # Perform meta-training step
                    prev_weights = torch.load(prev_model, map_location='cuda')
                    pred_gaze = model(image)
                    loss_gaze = F.l1_loss(pred_gaze, label)

                    model_grads = torch.autograd.grad(loss_gaze,
                                                      [p for p in model.parameters() if p.requires_grad],
                                                      allow_unused=True)

                    gradients = []
                    grad_counter = 0
                    for param in model.parameters():
                        if param.requires_grad:
                            gradient = model_grads[grad_counter]
                            grad_counter += 1
                        else:
                            gradient = None
                        gradients.append(gradient)

                    model_meta_grads = {name: g for ((name, _), g) in zip(model.named_parameters(), gradients)}
                    model.load_state_dict(prev_weights)

                    pred_gaze = model(dummy_input)
                    loss_gaze = F.l1_loss(pred_gaze, dummy_gt)
                    gen_gradients = model_meta_grads

                    util.meta_update_model(model, optimizer, loss_gaze, gen_gradients)
                    torch.save(model.state_dict(), prev_model)

            mae = util.evaluation_this_epoch(model, val_loader)
            logger.writerow({'epoch': str(epoch + 1).zfill(3), 'MAE': f'{mae:.3f}'})
            log.flush()

            is_best = mae < best
            if is_best:
                best = mae

            save = {'epoch': epoch, 'model': copy.deepcopy(model).half()}
            torch.save(save, f'weights/meta_last.pt')

            if is_best:
                torch.save(save, f'weights/meta_best.pt')
            del save

        util.strip_optimizer('weights/meta_best.pt')
        util.strip_optimizer('weights/meta_last.pt')
        torch.cuda.empty_cache()


def demo(args):
    import time
    model = nn.gaze_network()
    checkpoint = torch.load('weights/tem_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    util.freeze_params(model)
    util.make_padding_trainable(model)
    for name, w in model.named_parameters():
        if 'prompt' in name:
            w.requires_grad = True
    model.cpu().eval()
    detector = RetinaFace(0)

    stream = cv2.VideoCapture(0)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize])

    if not stream.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = stream.read()
        if not ret:
            break
        faces = detector(frame)

        for box, landmarks, score in faces:
            if score < .95:
                continue
            x_min, y_min = int(box[0]), int(box[1])
            x_max, y_max = int(box[2]), int(box[3])
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)

            x_min = max(0, x_min - int(0.2 * bbox_height))
            y_min = max(0, y_min - int(0.2 * bbox_width))
            x_max = x_max + int(0.2 * bbox_height)
            y_max = y_max + int(0.2 * bbox_width)

            image = frame[y_min:y_max, x_min:x_max, :]
            image = Image.fromarray(image)
            image = image.convert('RGB')
            image = transform(image)
            image = image.unsqueeze(0)
            image = image.cpu()
            start = time.time()
            pitch, yaw = model(image)
            end = time.time()
            print('FPS: {:.2f}'.format(1 / (end - start)))
            print(pitch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../../Datasets/Gaze/Gaze_360/')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=120)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_patience', type=int, default=25)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--meta_train', default=True, action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    os.makedirs('./weights', exist_ok=True)

    if args.train:
        train(args)
    if args.meta_train:
        meta_train(args)
    if args.demo:
        demo(args)


if __name__ == '__main__':
    main()


