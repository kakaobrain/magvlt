def convert_image_to_rgb(image):
    return image.convert("RGB")


def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip().lower() for l in lines]
    return lines