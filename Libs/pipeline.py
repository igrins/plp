from . import basic

if 1:




    def readechellogram(item_list):

        imgs = []
        hdrs = []
        for item in item_list:
            img, hdr = basic.readfits(item)
            #img_min = array(img).min()
            #print 'item', item
            #if img_min <= 0:
            #    img -= img_min

            imgs.append(img)
            hdrs.append(hdr)
        return imgs,hdrs
