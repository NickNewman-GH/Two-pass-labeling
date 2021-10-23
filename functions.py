import numpy as np
import matplotlib.pyplot as plt

def check(image, y, x):
    if not 0 <= x < image.shape[1]:
        return False
    if not 0 <= y < image.shape[0]:
        return False
    if image[y, x] != 0:
        return True
    return False

def neigh2(image, y, x):
    left = y, x - 1
    top = y - 1, x
    if not check(image, *left):
        left = None
    if not check(image, *top):
        top = None
    return left, top

def find(label, linked):
    j = label
    j = int(j)
    while linked[j] != 0:
        j = linked[j]
    return j

def union(label1, label2, linked):
    j = find(label1, linked)
    k = find(label2, linked)
    if j != k:
        linked[k] = j

def two_pass_labeling(binary_image):
    labeled = np.zeros_like(binary_image)
    label = 1
    linked = np.zeros(len(binary_image), dtype="uint")
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] != 0:
                ns = neigh2(binary_image, i, j)
                if ns[0] is None and ns[1] is None:
                    m = label
                    label += 1
                else:
                    lbs = [labeled[k] for k in ns if k is not None]
                    m = min(lbs)
                labeled[i, j] = m
                for n in ns:
                    if n is not None:
                        lbl = labeled[n]
                        if lbl != m:
                            union(m, lbl, linked)

    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] != 0:
                new_label = find(labeled[i, j], linked)
                if new_label != labeled[i, j]:
                    labeled[i, j] = new_label

    digits = np.unique(labeled.flatten())

    linked = {}

    num_of_digits = len(digits)

    for i in range(1,num_of_digits):
        linked[digits[i]] = i

    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] != 0:
                new_label = linked[labeled[i, j]]
                if new_label != labeled[i, j]:
                    labeled[i, j] = new_label
    return labeled

if __name__ == "__main__":
    B = np.zeros((20, 20), dtype='int32')
    
    B[1:-1, -2] = 1
    
    B[1, 1:5] = 1
    B[1, 7:12] = 1
    B[2, 1:3] = 1
    B[2, 6:8] = 1
    B[3:4, 1:7] = 1
    
    B[7:11, 11] = 1
    B[7:11, 14] = 1
    B[10:15, 10:15] = 1
    
    B[5:10, 5] = 1
    B[5:10, 6] = 1

    LB = two_pass_labeling(B)
    
    print("Labels - ", list(set(LB.ravel()))[1:])
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(B, cmap="hot")
    plt.colorbar(ticks=range(int(2)))
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(LB.astype("uint8"), cmap="hot")
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.show()
