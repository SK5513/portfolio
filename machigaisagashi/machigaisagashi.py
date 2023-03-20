import cv2
import numpy as np

# debug用画像の保存
def save_output_img(img , img_name, process) :
        path = "./debug_img/" + img_name +"_"+ process + ".png"
        cv2.imwrite(path, img)
        
        print("")
        print("Saved : ", path)


image_name = 'test'
# 元の画像読み込み
image = cv2.imread("./original_image/sample_5.jpg")

# マージンをつける
# 画像のサイズを取得する
height, width, channels = image.shape
margin_size = 30
new_height = height + margin_size * 2
new_width = width + margin_size * 2

new_image = np.ones((new_height, new_width, channels), np.uint8)*255
x_offset = margin_size
y_offset = margin_size
new_image[y_offset:y_offset+height, x_offset:x_offset+width] = image

save_output_img(new_image,image_name,'margin')

# グレースケール化
gray_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
save_output_img(gray_image,image_name,'gray')
#　2値化
bin_image = cv2.adaptiveThreshold(gray_image,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,39,3)
save_output_img(bin_image,image_name,'bin')

#　矩形の輪郭抽出
contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 面積が大きい順にソートし、2つの輪郭を取得
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

#　二つの矩形の頂点座標を取得
epsilon_a = 0.05*cv2.arcLength(contours[0],True)
approx_a = cv2.approxPolyDP(contours[0],epsilon_a,True)
epsilon_b = 0.05*cv2.arcLength(contours[1],True)
approx_b = cv2.approxPolyDP(contours[1],epsilon_b,True)

#print(approx_0)
#print(approx_1)

#二つの矩形画像を切り取る
A_x1,A_y1 =np.hsplit(approx_a[0],[1])
A_x2,A_y2 =np.hsplit(approx_a[2],[1])
B_x1,B_y1 =np.hsplit(approx_b[0],[1])
B_x2,B_y2 =np.hsplit(approx_b[2],[1])

imageA = new_image[int(A_y1):int(A_y2), int(A_x1):int(A_x2)]
save_output_img(imageA,image_name,'A')
imageB = new_image[int(B_y1):int(B_y2), int(B_x1):int(B_x2)]
save_output_img(imageB,image_name,'B')
'''
#二つの矩形の座標を取得
rect1 = cv2.boundingRect(contours[0])
rect2 = cv2.boundingRect(contours[1])

#二つの矩形画像を切り取る
x1, y1, w1, h1 = rect1
x2, y2, w2, h2 = rect2

imageA = image[y1:y1+h1, x1:x1+w1]
imageB = image[y2:y2+h2, x2:x2+w2]
'''

