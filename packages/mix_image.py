#!/usr/bin/env python
# coding: utf-8

# In[12]:

from PIL import Image


#@profile
def mix_images(img,logo,OUTPUT=None,input_type='image'):

    #route
    if input_type=='image':
        img = Image.open(img)
        logo = Image.open(logo)
    #array
    #just do it
    
    layer = Image.new('RGBA', img.size, (255,255,255,0))
    layer.paste(logo, (img.size[0]-logo.size[0], img.size[1]-logo.size[1]))
    img_after = Image.composite(layer, img, layer)
    
    #route
    if input_type=='image':
        img_after.save(OUTPUT)
        img_after.close()
    else:
    #just do it
        return img_after
    
    img.close
    logo.close
    layer.close
    del layer,logo,img,OUTPUT,img_after

'''
class mix_images:
    
    @profile
    def __init__(self, imgIn, logoIn, OUTPUT):
        self.img = Image.open(imgIn)
        self.logo = Image.open(logoIn)
        self.layer = Image.new('RGBA', self.img.size, (255,255,255,0))
        self.layer.paste(self.logo, (self.img.size[0]-self.logo.size[0], self.img.size[1]-self.logo.size[1]))
        self.img_after = Image.composite(self.layer, self.img, self.layer)
        self.img_after.save(OUTPUT)
        
        #del layer,logo,img,OUTPUT#,img_after
        #gc.collect()
        print('finir')
        print('finir')
        print('finir')

    @profile
    def __del__(self):
        self.img_after.close()
        self.img.close()
        self.logo.close()
        self.layer.close()

@profile
def funa(a, b, c):
    for i in range(0, 9):
        test = mix_images(a,b,c)
        del test
'''

#@profile
def func(a, b, c):
    for i in range(0, 9):
        mix_images(a,b,c)
        #del mix_images

if __name__ == "__main__":
    a="./NxmXK9.jpg"
    b="./15.png"
    c="./mix.png"
    #mix_images(a,b,c)
    func(a, b, c)
