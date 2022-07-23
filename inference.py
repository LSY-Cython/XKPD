import cv2
import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def preprocess(inp):

    inp = inp[:,:,::-1]
    inp = inp.astype(np.float32)
    inp = (inp-127.5)/128.0
    inp = inp.transpose((2,0,1))
    return inp

def decode(output):
    '''
    make sure output.shape is [B,N,W,H]
    '''
    batch_size,num_joints = output.shape[:2]
    height,width = output.shape[2:]
    output = output.reshape(batch_size,num_joints,-1)
    output_numpy = output.detach().cpu().numpy()

    idx = np.argmax(output_numpy,2)
    maxvals = np.amax(output_numpy,2)
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask

    return preds

def view(img, kps, flag='single'):
    # img = Image.open(path)
    # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img = cv2.imread(path)
    for i in range(kps.shape[0]):
        kp_x,kp_y = kps[i,:2]
        int_kpx,int_kpy = int(kp_x),int(kp_y)
        cv2.putText(img,'%d'%(i),(int_kpx+5,int_kpy-5),cv2.FONT_HERSHEY_SIMPLEX,1.5, (232, 9, 52), 4)
        cv2.circle(img,(int_kpx,int_kpy),15, (38, 46, 222), -1)
    return img

# def view(path,kps,flag='single'):
#     img = Image.open(path)
#     img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
#     # img = cv2.imread(path)
#     for i in range(kps.shape[0]):
#         kp_x,kp_y = kps[i,:2]
#         int_kpx,int_kpy = int(kp_x),int(kp_y)
#         cv2.putText(img,'%d'%(i),(int_kpx+5,int_kpy-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255), 4)
#         cv2.circle(img,(int_kpx,int_kpy),14, (0, 255, 0), -1)
    # contour = np.expand_dims(kps, axis=0).astype(np.int32)
    # contours返回格式为包含多个np.int32轮廓的列表[array([[[a,b],[c,d],...,[x,y]]]), array([[[a,b],[c,d],...,[x,y]]]), ...]
    # 按每个轮廓的组成点顺序连接, -1表示画出所有轮廓
    # img = cv2.drawContours(img, [contour], -1,(0,255,0),5)

    # if flag == 'single':
    #     basename = path.split('.')[0]
    #     # cv2.imwrite('result_point.jpg', img)
    # else:
    #     dirname,basename = os.path.dirname(path),os.path.basename(path)
    #     if not os.path.exists('%s_Finished'%(dirname)):
    #         os.makedirs('%s_Finished'%(dirname))
    #     # cv2.imwrite('%s_Finished/%s'%(dirname,basename),img)
    # return img

# def _imread(path):
#
#     IMAGE_SIZE = (256,256)
#     raw_img = Image.open(path)
#     # raw_img = cv2.imread(path)
#     raw_img = cv2.cvtColor(np.asarray(raw_img), cv2.COLOR_RGB2BGR)
#     assert raw_img is not None,'%s is not imread'%(path)
#     img_h,img_w = raw_img.shape[:2]
#     img = cv2.resize(raw_img,IMAGE_SIZE)
#     scale = (img_h/IMAGE_SIZE[0],img_w/IMAGE_SIZE[1])
#     inp = preprocess(img)
#     return inp,scale

def _imread(raw_img):

    IMAGE_SIZE = (256,256)
    # raw_img = Image.open(path)
    # raw_img = cv2.imread(path)
    # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
    # assert raw_img is not None,'%s is not imread'%(path)
    img_h,img_w = raw_img.shape[:2]
    img = cv2.resize(raw_img,IMAGE_SIZE)
    scale = (img_h/IMAGE_SIZE[0],img_w/IMAGE_SIZE[1])
    inp = preprocess(img)
    return inp,scale

def handle_single(raw_img,model,device):

    # inp,scale = _imread(path)
    inp, scale = _imread(raw_img)
    with torch.no_grad():
        inp = torch.tensor([inp]).to(device)
        output = model(inp)     # (B,44,64,64)
        kps = decode(output)    # (B,44,2)
        kps = kps*4
        kps[:,:,::2] *= scale[1]
        kps[:,:,1::2] *= scale[0]

        kps = kps.squeeze(0)#*4#*max(scale)
        return view(raw_img.copy(),kps), kps

def handle_file(filename,model,device):
    img_list_ = os.listdir(filename)
    img_list = [os.path.join(filename,item) for item in img_list_]
    batch = 8

    inps = []
    scales = []
    items = []
    for i,item in enumerate(img_list):
        if (i+1) % batch!=0:
            inp,scale = _imread(item)
            scales.append(scale)
            inps.append(inp)
            items.append(item)
        else:
            inp,scale = _imread(item)
            scales.append(scale)
            inps.append(inp)
            items.append(item)
            with torch.no_grad():
                inps = torch.tensor(inps,dtype=torch.float32).to(device)
                scales = np.array(scales,dtype=np.float32)
                output = model(inps)     # (B,44,64,64)
                kps = decode(output)    # (B,44,2)
                kps = kps*4

                for i in range(kps.shape[0]):
                    kps[i,:,::2] *= scales[i,1]
                    kps[i,:,1::2] *= scales[i,0]
                    view(items[i],kps[i],flag='file')
            inps = []
            scales = []
            items = []
    if len(inps):
        with torch.no_grad():
            inps = torch.tensor(inps,dtype=torch.float32).to(device)
            scales = np.array(scales,dtype=np.float32)
            output = model(inps)     # (B,44,64,64)
            kps = decode(output)    # (B,44,2)
            kps = kps*4

            for i in range(kps.shape[0]):
                kps[i,:,::2] *= scales[i,1]
                kps[i,:,1::2] *= scales[i,0]
                view(items[i],kps[i],flag='file')


if __name__ == '__main__':
    # print(torch.cuda.device_count())
    gpu_flag = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    test_path = 'test/1.jpg'
    demo_file = 'test'
    pth = 'output/model-lr-050.pth'
    
    if gpu_flag:
        device = 'cuda'
        model = torch.load('net.pth')
        state_dict = torch.load(pth)
    else:
        device = 'cpu'
        model = torch.load('net.pth',map_location='cpu')
        state_dict = torch.load(pth,map_location='cpu')
    model.load_state_dict({k.replace('module.',''):v for k,v in state_dict['model'].items()})
    model = model.to(device)
    model.eval()

    handle_single(test_path,model,device)
    # handle_file(demo_file,model,device)

   







