import numpy as np
from visualizer import Visualizer
import matplotlib.pyplot as plt


def inference_sam_m2m_auto(image, outputs, label_mode='1', alpha=0.1, anno_mode=['Mask']):
    image = np.asarray(image)
    visual = Visualizer(image)
    sorted_anns = sorted(outputs, key=(lambda x: x['area']), reverse=True)
    label = 1

    mask_map = np.zeros(image.shape, dtype=np.uint8)    
    for i, ann in enumerate(sorted_anns):
        mask = ann['segmentation']
        demo = visual.draw_binary_mask_with_number(mask, text=str(label), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
        mask_map[mask == 1] = label
        label += 1
    im = demo.get_image()
    return im, sorted_anns

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))