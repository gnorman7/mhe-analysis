import matplotlib.pyplot as plt

def plot_img(img):
    fig, ax = plt.subplots(figsize = (8, 8))
    ax.imshow(img)
    ax.axis('off')
    return fig, ax

