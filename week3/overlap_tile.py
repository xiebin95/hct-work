import torch
import cv2

def overlap_tile(img, padding_x=20, padding_y=20):
    # Pad top and bottom
    top_pad = torch.flip(img[:padding_y], dims=[0]) # Get slice and flip
    bot_pad = torch.flip(img[-padding_y:], dims=[0])
    img = torch.cat([top_pad, img, bot_pad], dim=0) # Concatenate tensors
    # Pad left and right
    left_pad = torch.flip(img[:, :padding_x], dims=[1])
    right_pad = torch.flip(img[:, -padding_x:], dims=[1])
    img = torch.cat([left_pad, img, right_pad], dim=1)
    return img


if __name__ == '__main__':

    # Read image
    img = cv2.imread('test.jpg')
    if img is None:
        print('No such image file')
        exit()

    # Conver numpy.array to torch.Tensor 
    img = torch.from_numpy(img)

    # Conduct overlap tile strategy
    img = overlap_tile(img, padding_x=img.shape[1]//4, \
                            padding_y=img.shape[0]//4)

    # Conver torch.Tensor to numpy.array
    img = img.numpy()

    # Save image
    cv2.imwrite('overlap_tile.jpg', img)
