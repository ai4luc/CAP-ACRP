import numpy as np
import matplotlib.pyplot as plt

def plot_rgb(r, g, b, scale_min_max=True, stretch_factor=1, normalize_data=False):
    # Functions for contrast enhancements
        # Rescale to 0 and 1 using minimum and maximum values
    def scaleMinMax(x, factor=1):
        return((x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x))) * factor
    
    def normalize(x, factor=1):
        return(x/255) * factor

    if scale_min_max is True:
        r = scaleMinMax(r, stretch_factor)
        g = scaleMinMax(g, stretch_factor)
        b = scaleMinMax(b, stretch_factor)
    elif normalize_data is True:
        r = normalize(r, stretch_factor)
        g = normalize(g, stretch_factor)
        b = normalize(b, stretch_factor)

    rgbMinMax = np.dstack((r, g, b))

    plt.imshow(rgbMinMax)
    plt.axis("off");
    plt.show()