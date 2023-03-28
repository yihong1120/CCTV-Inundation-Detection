import PIL.Image as Image


def transparent_back(r, g, b, a, imgn):
    img = imgn.convert('RGBA')
    data = img.getdata()
    new_data = [(r, g, b, 0) if pixel == (r, g, b, a) else pixel for pixel in data]
    img.putdata(new_data)
    return img


if __name__ == "__main__":
    img_path = "./14.png"
    output_path = "./14_out.png"
    img = Image.open(img_path)
    for i in range(3):
        img = transparent_back(0, 0, 0, 255, img)
        img.save(output_path)
    img.close()
