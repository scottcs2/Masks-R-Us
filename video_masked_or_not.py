import cv2
import torch
from torch import nn
from torchvision import models, transforms

classes = ['Masked', 'Masked Improperly', 'Not Masked']

#model_path = 'best_model_0.8919191360473633.pt'
#model_path = 'incomplete_model_0.9666666388511658.pt'
# = 'augmented_model_0.6538333296775818.pt'
#model_path = 'best_model_0.9816998839378357.pt'
#model_path = 'in_progress_m_or_n_model_0.8888888955116272.pt'
#model_path = 'in_progress_m_or_n_model_0.855555534362793.pt' # working half decently with some adjustment
model_path = 'Working_m_or_n.pt'

pre_process_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main():
    video_capture, net = setupDetection()

    while True:
        detectMasks(video_capture, net)

        if qPressed():
            break

def setupDetection():
    video_capture = cv2.VideoCapture(0)
    net = load_model()
    return video_capture, net
    

def detectMasks(video_capture, net):
    # Read in video from the webcam
    _, frame = video_capture.read()
    
    # Use the ResNet to predict a label
    label = get_label(frame, net)
    
    
    # Add text with label to frame
    put_label_on_frame(frame,label)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)

def qPressed():
    return cv2.waitKey(15) & 0xFF == ord('q')
    
    
    
def load_model():
    
    
    device = torch.device('cpu')
    
    
    # Get a resnet 50 model  
    net = models.resnet50()
    
    
    # Modify the last layer of that net for three classes
    in_features = net.fc.in_features
    hidden_size = 500 # 500# 256
    num_classes = 3 # mask, improper mask, no mask
    
    classifier = nn.Sequential(
        nn.Linear(in_features, hidden_size),
        nn.ReLU(),
        nn.Dropout(.2),
        nn.Linear(hidden_size, num_classes)
    )
    
    # Update the last layer to be several classifications layers
    net.fc = classifier
    
    net.load_state_dict(torch.load(model_path, map_location=device))
    
    net.eval()
    
    return net
    
def get_label(frame, net):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    img = pre_process(frame)
    
    out = net(img)
    
    # Take softmax of out for %
    out = out.exp()
    out /= out.sum()
    
    # This adjustment works well in practice
    out[0,2] += .11 # .05 # .07 a little better
    out[0,0] -= .11
    
    val, pred = torch.max(out, dim=1)
    
    val = val.item()
    pred = pred.item()
    
    threshold = 0
    
    print(out)
    
    if pred == 1:
        pred = 0
    
    if val > threshold:
        label = classes[pred]
        
    else:
        label = 'Unknown'
    
    return label

def pre_process(frame):
    # Frame is 720x1280x3 and is int values to 255
    # We want to convert it to 1x3x256x256 where values are 0 to 1 and normalized by imagenet mean and std
    img = torch.tensor(frame)

    img = img.unsqueeze(0).permute(0,3,1,2).to(dtype=torch.float32) / 255.0
    
    #print(img)
    
    img = pre_process_transforms(img)
    
    return img
    
def put_label_on_frame(frame,label):
    cv2.putText(frame,  
                label,  
                (50, 50),  
                cv2.FONT_HERSHEY_SIMPLEX, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
    

if __name__ == '__main__':
    main()