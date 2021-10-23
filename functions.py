import numpy as np

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
                ns = neigh2(image, i, j)
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
