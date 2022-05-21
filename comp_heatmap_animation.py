import os

import matplotlib.animation as animation
import matplotlib.image as mgimg
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():

    model_name = "Attention Only"
    file_dir = f'{model_name}/composition_heatmaps'
    
    img_files = sorted([f for f in os.listdir(file_dir) if f.endswith(".jpg")], key = lambda x: int(x.split('_')[1].split('.')[0]))
    
    images = []
    fig = plt.figure(figsize=(12,4))
    for f in tqdm(img_files):
        image = mgimg.imread(os.path.join(file_dir, f))
        images.append([plt.imshow(image)])
    plt.axis('off')
    my_anim = animation.ArtistAnimation(fig, images, interval=20, repeat_delay=2000)
    my_anim.save(f'{model_name}/composition_heatmaps.gif', fps=10)

if __name__ == "__main__":
    main()
