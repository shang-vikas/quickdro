import cv2
import torch
import numpy as np
import pandas as pd
import gc

# https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb-0-892
BASE_SIZE = 256
colors = [(255, 0, 0) , (255, 255, 0),  (128, 255, 0),  (0, 255, 0), (0, 255, 128), (0, 255, 255), 
          (0, 128, 255), (0, 0, 255), (128, 0, 255), (255, 0, 255)]

def list2drawing(raw_strokes, size=64, lw=None, time_color=False):
    img = np.zeros((BASE_SIZE, BASE_SIZE,3), np.uint8)
    lw = lw if lw is not None else np.random.choice([2,4,6])
    strks = np.random.choice([0,1],len(raw_strokes),p=[0.1,0.9])
    for t, stroke in enumerate(raw_strokes):
        if strks[t] ==0:
            continue
        for i in range(len(stroke[0]) - 1):
            color = colors[min(len(colors)-1, i)] if time_color else (255,0,255)
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw, lineType=cv2.LINE_AA)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        return img

def drawing2tensor(drawing):
    # rgb = cv2.cvtColor(drawing,cv2.COLOR_GRAY2RGB)
    rgb = drawing.transpose(2,0,1).astype(np.float32)
    return torch.from_numpy(rgb)

# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def map3(preds, targs):
    predicted_idxs = preds.sort(descending=True)[1]
    top_3 = predicted_idxs[:, :3]
    res = mapk([[t] for t in targs.cpu().numpy()], top_3.cpu().numpy(), 3)
    return torch.tensor(res)

def top_3_preds(preds): return np.argsort(preds.numpy())[:, ::-1][:, :3]

def top_3_pred_labels(preds, classes):
    top_3 = top_3_preds(preds)
    labels = []
    for i in range(top_3.shape[0]):
        labels.append(' '.join([classes[idx] for idx in top_3[i]]))
    return labels


def focal_loss(input,targets,alpha=1,gamma=1,reduce=True):        
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
    pt = torch.exp(-BCE_loss)
    F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

    if self.reduce:
        return torch.mean(F_loss)
    else:
        return F_loss





def create_submission(test_preds, test_dl, name, classes):
    key_ids = [path.stem for path in test_dl.dataset.x.items]
    labels = top_3_pred_labels(test_preds, classes)
    sub = pd.DataFrame({'key_id': key_ids, 'word': labels})
    sub.to_csv(f'subs/{name}.csv.gz', index=False, compression='gzip')
