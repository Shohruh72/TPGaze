import torch
import numpy as np
from nets import nn


def model_info(model):
    print('[*] Number of model parameters: {:,}'.format(sum([p.data.nelement() for p in model.parameters()])))
    print('[*] Number of trainable model parameters: {:,}'.format(
        sum([p.data.nelement() if p.requires_grad else 0 for p in model.parameters()])))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def strip_optimizer(filename):
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def gaze3d(gaze):
    n = gaze.shape[0]
    sin = np.sin(gaze)
    cos = np.cos(gaze)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 1], sin[:, 0])
    out[:, 1] = sin[:, 1]
    out[:, 2] = np.multiply(cos[:, 1], cos[:, 0])
    return out


def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = gaze3d(a) if a.shape[1] == 2 else a
    b = gaze3d(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * 180.0 / np.pi


def freeze_params(model):
    for c in model.children():
        for param in c.parameters():
            param.requires_grad = False


def make_padding_trainable(model, crop_size=224):
    c_count = 0
    m_count = 0
    for i, m in enumerate(model.gaze_network.children()):
        if i < 6:
            if isinstance(m, nn.Conv2d):
                c_count += 1
                P = m.padding[0]
                S = m.stride[0]
                K = m.kernel_size[0]
                m.data_crop_size = crop_size
                m.make_padding_trainable()
                for k, v in m.state_dict().items():
                    if k in ['prompt_embeddings_tb', 'prompt_embeddings_lr']:
                        for param in v:
                            param.requires_grad = True
                crop_size = (crop_size + 2 * P - K) // S + 1
            elif isinstance(m, torch.nn.MaxPool2d):
                m_count += 1
                P = m.padding
                S = m.stride
                K = m.kernel_size
                crop_size = (crop_size + 2 * P - K) // S + 1
            elif isinstance(m, torch.nn.Sequential):
                for b in m.children():
                    for c in b.children():
                        if isinstance(c, nn.Conv2d):
                            if c.padding:
                                c_count += 1
                                P = c.padding[0]
                                S = c.stride[0]
                                K = c.kernel_size[0]
                                c.data_crop_size = crop_size
                                c.make_padding_trainable()
                                for k, v in c.state_dict().items():
                                    if k in ['prompt_embeddings_tb', 'prompt_embeddings_lr']:
                                        for param in v:
                                            param.requires_grad = True
                                crop_size = (crop_size + 2 * P - K) // S + 1


def meta_update_model(model, optimizer, loss, gradients):
    hooks = []
    for (k, v) in model.named_parameters():
        def get_closure():
            key = k

            def replace_grad(grad):
                return gradients[key]

            return replace_grad

        if gradients[k] is not None:
            hooks.append(v.register_hook(get_closure()))

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    for h in hooks:
        h.remove()


def evaluation_this_epoch(model, loader):
    import tqdm
    model.eval()
    errors = AverageMeter()

    for input_img, label in tqdm.tqdm(loader, '%10s' % 'MAE'):
        input_var = input_img["face"].cuda()
        pred_gaze = model(input_var)

        gaze_error_batch = np.mean(angular_error(pred_gaze.cpu().data.numpy(), label.data.numpy()))
        errors.update(gaze_error_batch.item(), input_var.size()[0])

    print('%10s' % f'{errors.avg:.3f}')
    model.float()  # for training
    return errors.avg


def save_checkpoint(state, add=None):
    import os
    if add is not None:
        filename = add + '_ckpt.pth.tar'
    else:
        filename = 'ckpt.pth.tar'
    ckpt_path = os.path.join('weights', filename)
    torch.save(state, ckpt_path)

    print('save file to: ', filename)
