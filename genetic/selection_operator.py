from __future__ import division
import numpy as np
class Selection(object):

    def RouletteSelection(self, _a, k):
        a=np.sort(_a)
        a = np.asarray(a)
        b=[]
        c=[]
        for i in range(len(a)):
            if a[i] <= 0 :
                b.append(a[i])
            else:
                c.append(a[i])
        b=np.asarray(b)
        e_x = np.exp(b - np.max(b))
        b = e_x / e_x.sum()
        c = np.asarray(c)
        a = np.append(b,c)
        idx = np.argsort(a)
        idx = idx[::-1]
        sort_a = a[idx]
        sum_a = np.sum(a).astype(np.float)
        selected_index = []
        for i in range(k):
            u = np.random.rand()*sum_a
            print("u",u)
            sum_ = 0
            for j in range(sort_a.shape[0]):
                sum_ +=sort_a[j]
                if sum_ > u:
                    selected_index.append(idx[j])
                    break

        return selected_index


if __name__ == '__main__':
    s = Selection()
    a = [1, 3, 2, 1, 4, 4, 5]
    selected_index = s.RouletteSelection(a, k=20)

    new_a =[a[i] for i in selected_index]
    print(list(np.asarray(a)[selected_index]))
    print(new_a)






