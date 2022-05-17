from torchvision.transforms import InterpolationMode, Resize
import numpy as np
import cv2
from io import BytesIO
from PIL import Image


def improve_img(image, shape = None, brightness = 1): # image.prop = torch.Tensor(3, , )
    if shape is not None:
        image = Resize((shape), interpolation=InterpolationMode.BICUBIC)(image)

    image = image.detach()[0].permute(1, 2, 0).numpy()
    kernel = brightness*np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return image_sharp


def download_image(image):
    dimg = image*255
    dimg = Image.fromarray(dimg.astype(np.uint8))
    buf = BytesIO()
    dimg.save(buf, format="PNG")
    return buf.getvalue()