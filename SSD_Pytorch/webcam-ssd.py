import torch 
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import numpy as np

def detect(frame, net, transform, device):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2,0,1)
    x = Variable(x.unsqueeze(0)).to(device)
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            if not(np.isnan(pt).any() or np.isinf(pt).any() or \
                   (pt > max(width, height)).any() or (pt < 0).any()):
                cv2.rectangle(frame, (int(pt[0]), int(pt[1])), 
                 (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
                cv2.putText(frame, labelmap[i-1], (int(pt[0]), int(pt[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 
                2, cv2.LINE_AA)
            j += 1
        if i == 15:
            print('cantidad de personas: %d' % j)

net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',
                               map_location = lambda storage, loc: storage))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    detect(frame, net, transform, device)
    cv2.imshow('Object Recognition', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

