"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import torch as th
import torch.distributed as dist
import torchvision as tv

from PIL import Image
import numpy as np

from guided_diffusion.image_datasets import load_data

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import importlib


def visualize_cond(cond, hv_map=False, inst=False, col_map=False, out_path=None):
    out = []
    tmp = cond.cpu().numpy().transpose((1, 2, 0))
    if hv_map:
        hv_map = tmp[..., :2]
        tmp = tmp[..., 2:]
        targets = importlib.import_module('hover_net.models.hovernet.targets')
        vis_hv_map = getattr(targets, 'vis_hv_map')
        hv_map = vis_hv_map(hv_map)
        out.append(hv_map[:, :256, :])
        out.append(hv_map[:, 256:, :])
    if col_map:
        col_map = (tmp[..., :3] +1 ) * 127.5
        tmp = tmp[..., 3:]
        out.append(col_map.astype(np.uint8))
    if inst:
        edge_map = tmp[..., :1]*255
        out.append(edge_map.repeat(3, axis=-1).astype(np.uint8))

        tmp = tmp[..., 1:]

    out = np.concatenate(out, axis=1)

    if out_path is not None:
        Image.fromarray(out).save(out_path)

    return out


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    data = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False,
        random_crop=False,
        random_flip=False,
        is_train=args.use_train,
        num_classes=args.num_classes,
        use_hv_map=args.use_hv_map,
        use_col_map=args.use_col_map,
        preserve_nuclei_col=args.preserve_nuclei_col,
        in_channels=args.in_channels,
    )

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    image_path = os.path.join(args.results_path, 'images')
    os.makedirs(image_path, exist_ok=True)
    label_path = os.path.join(args.results_path, 'labels')
    os.makedirs(label_path, exist_ok=True)
    cond_path = os.path.join(args.results_path, 'cond')
    os.makedirs(cond_path, exist_ok=True)
    sample_path = os.path.join(args.results_path, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    if not args.no_instance:
        inst_path = os.path.join(args.results_path, 'inst_masks')
        os.makedirs(inst_path, exist_ok=True)


    logger.log("sampling...")
    all_samples = []
    for i, (batch, cond) in enumerate(data):
        image = ((batch + 1.0) / 2.0).cuda()
        label = (cond['label_ori'].float() / 255.0).cuda()
        model_kwargs = preprocess_input(cond, num_classes=args.num_classes, class_cond=args.class_cond)

        # set hyperparameter
        model_kwargs['s'] = args.s

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels, image.shape[2], image.shape[3]),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True
        )
        sample = (sample + 1) / 2.0

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        if 'instance' in cond:
            inst_map = cond['instance'].cpu().numpy().astype(np.uint16)
            for j in range(inst_map.shape[0]):
                Image.fromarray(inst_map[j].squeeze()).save(os.path.join(inst_path, cond['path'][j].split('/')[-1].split('.')[0] + '.tif'))        

        for j in range(sample.shape[0]):
            img_name = cond['path'][j].split('/')[-1].split('.')[0] + '.png'
            tv.utils.save_image(image[j], os.path.join(image_path, img_name))
            tv.utils.save_image(sample[j], os.path.join(sample_path, img_name))
            tv.utils.save_image(label[j], os.path.join(label_path, img_name))

            visualize_cond(model_kwargs['y'][j], hv_map=args.use_hv_map, inst=not args.no_instance, col_map=args.use_col_map, 
                           out_path=os.path.join(cond_path, img_name))

        logger.log(f"created {len(all_samples) * args.batch_size} samples")

        if len(all_samples) * args.batch_size > args.num_samples:
            break

    dist.barrier()
    logger.log("sampling complete")


def preprocess_input(data, num_classes, class_cond=True):
    if class_cond:
        # move to GPU and change data types
        # data['label'] = data['label'].long()
        # create one-hot label map
        label_map = data['label'].long()
        bs, _, h, w = label_map.size()
        nc = num_classes
        input_label = th.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

    else:
        input_semantics = data['label']

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

    return {'y': input_semantics}


def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        use_hv_map=False,
        use_col_map=False,
        preserve_nuclei_col=False,
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        results_path="",
        is_train=False,
        s=1.0,
        no_instance=False,
        use_train=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
