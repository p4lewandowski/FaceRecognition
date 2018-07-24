import numpy as np

a = [5, 6, 14, 15, 3]
n = [0, 1, 3, 1, 0]

def sum_class_distances(distances, class_labels):
    un_val = np.unique(class_labels)
    arr = []
    for i in un_val:
        sum = 0
        count = 0;
        arr_pos = 0;
        for elem in class_labels:
            if elem==i:
                sum +=distances[arr_pos]
                count+=1
            arr_pos+=1
        arr.append(np.vstack((i, sum/count)))

    return arr

print(sum_class_distances(a, n))
