import PIL.Image as Image
import os

#@profile
def transparent_back(r,g,b,a,imgn,out_img=None, input_type='image'):

    #route
    if input_type=='image':
        imgn=Image.open(imgn)
        imgn = imgn.convert('RGBA')
    
    L, H = imgn.size
    color_0 = (r,g,b,a)#要替换的颜色
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = imgn.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                imgn.putpixel(dot,color_1)
                
    #route
    if input_type=='image':
        imgn.save(out_img)
    else:
        return imgn
    
    imgn.close()
    del imgn, L, H, color_0, color_1,r,g,b,a,out_img

if __name__ == "__main__":
    img="./14.png"
    output="./14_out.png"
    for i in range(3):
        transparent_back(0,0,0,255,img,output)
