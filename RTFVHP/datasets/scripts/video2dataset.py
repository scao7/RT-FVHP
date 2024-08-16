import cv2
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import romp
from romp import ROMP
import numpy as np
import json
from RTFVHP.humanModel.smpl.smpl_numpy import SMPL
from tqdm import tqdm
import pickle
import sys

MODEL_DIR='RTFVHP/humanModel/smpl-meta'
# Function to extract frames
def extract_frames(video_path, output_folder, num_frames=300):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Capture the video
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the interval between frames to extract
    frame_interval = total_frames // num_frames
    
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened() and extracted_count < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f'{extracted_count:06d}.jpg')
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f'Extracted {extracted_count} frames.')


# Function to apply DeepLabV3 and create a mask
def create_mask(image_path, output_path):

    # Define the transformation
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess the image
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Move the input to the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_batch = input_batch.to(device)

    # Perform the forward pass
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # Create a mask for the person class (class 15 in COCO dataset used by DeepLabV3)
    person_class = 15
    mask = output_predictions == person_class

    # Convert mask to uint8
    mask = mask.byte().cpu().numpy()

    # Save the mask
    mask_image = Image.fromarray(mask * 255)
    mask_image.save(output_path)

def rompEstimation(input_path,focal_length,imsize):
    print("estimating the romp parameters")
    settings = romp.main.default_settings 
    # settings is just a argparse Namespace. To change it, for instance, you can change mode via
    # settings.mode='video'
    romp_model = ROMP(settings)
    

    # crate humannerf metadata based on the romp estimation
    metadata = {}
    for image_path in sorted(os.listdir(input_path)):
        image_path_noextension = image_path.split(".")[0]
        # processing the images 
        outputs = romp_model(cv2.imread(os.path.join(input_path,image_path))) # please note that we take the input image in BGR format (cv2.imread).
        metadata[image_path_noextension] = {}
        metadata[image_path_noextension]["poses"] = outputs["smpl_thetas"].tolist()[0]
        metadata[image_path_noextension]["betas"] = outputs["smpl_betas"].tolist()[0]
        metadata[image_path_noextension]["cam_intrinsics"] = [
        [focal_length, 0.0,imsize[0]//2], 
        [0.0, focal_length, imsize[1]//2 ],
        [0.0, 0.0, 1.0]
        ]
        metadata[image_path_noextension]["cam_extrinsics"] = np.eye(4)
        metadata[image_path_noextension]["cam_extrinsics"][:3, 3] = outputs["cam_trans"][0]
        metadata[image_path_noextension]["cam_extrinsics"] = metadata[image_path_noextension]["cam_extrinsics"].tolist() 
        # https://github.com/Arthur151/ROMP/issues/300 cam extrinsic calculation
    return metadata

# preparedata as humannerf style
def prepareData(subject_dir):
    output_path=subject_dir
    with open(os.path.join(subject_dir, 'metadata.json'), 'r') as f:
        frame_infos = json.load(f)
    sex = 'neutral'
    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)
    cameras = {}
    mesh_infos = {}
    all_betas = []
    for frame_base_name in tqdm(frame_infos):
        cam_body_info = frame_infos[frame_base_name] 
        poses = np.array(cam_body_info['poses'], dtype=np.float32)
        betas = np.array(cam_body_info['betas'], dtype=np.float32)
        K = np.array(cam_body_info['cam_intrinsics'], dtype=np.float32)
        E = np.array(cam_body_info['cam_extrinsics'], dtype=np.float32)
        
        all_betas.append(betas)

        ##############################################
        # Below we tranfer the global body rotation to camera pose

        # Get T-pose joints
        _, tpose_joints = smpl_model(np.zeros_like(poses), betas)

        # get global Rh, Th
        pelvis_pos = tpose_joints[0].copy()
        Th = pelvis_pos
        Rh = poses[:3].copy()

        # get refined T-pose joints
        tpose_joints = tpose_joints - pelvis_pos[None, :]

        # remove global rotation from body pose
        poses[:3] = 0

        # get posed joints using body poses without global rotation
        _, joints = smpl_model(poses, betas)
        joints = joints - pelvis_pos[None, :]

        mesh_infos[frame_base_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses,
            'joints': joints,
            'tpose_joints': tpose_joints
        }

        cameras[frame_base_name] = {
            'intrinsics': K,
            'extrinsics': E
        }

    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)
        
    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:   
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    _, template_joints = smpl_model(np.zeros(72), avg_betas)
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:   
        pickle.dump(
            {
                'joints': template_joints,
            }, f)

def getSMPLmask(image_folder,output_path):
    # extract_frames(video_path, image_folder)
    settings = romp.main.default_settings 
    settings.mode = 'image'
    settings.render_mesh=True
    settings.cal_smpl = True
    settings.show_largest = True
    settings.show_items='smpl_mask'
    settings.renderer = 'sim3dr'
    settings.show_largest=True
    romp_model = romp.ROMP(settings)
    for i in tqdm(os.listdir(image_folder)):
        outputs = romp_model(cv2.imread(os.path.join(image_folder,i)))
        # Save the mask
        mask_image = Image.fromarray(np.asarray(outputs['smpl_mask']))
        mask_image.save(os.path.join(output_path, i[:-4] + '.png'))


def combine_mask(mask_folder,smpl_mask_folder,output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of files in both folders
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])
    smpl_mask_files = sorted([f for f in os.listdir(smpl_mask_folder) if f.endswith('.png')])

    # Check if both folders have the same number of files
    if len(mask_files) != len(smpl_mask_files):
        print("The number of files in the two folders do not match.")
        exit()
    # Combine masks using bitwise OR
    for mask_file, smpl_mask_file in zip(mask_files, smpl_mask_files):
        mask_path = os.path.join(mask_folder, mask_file)
        smpl_mask_path = os.path.join(smpl_mask_folder, smpl_mask_file)
        
        mask = Image.open(mask_path).convert('L')
        smpl_mask = Image.open(smpl_mask_path).convert('L')
        
        # Convert images to numpy arrays
        mask_array = np.array(mask)
        smpl_mask_array = np.array(smpl_mask)
        
        # Perform bitwise OR operation
        combined_mask_array = np.bitwise_or(mask_array, smpl_mask_array)
        
        # Convert the result back to an image
        combined_mask = Image.fromarray(combined_mask_array)
        
        # Save the result
        combined_mask.save(os.path.join(output_folder, mask_file))

    print("Masks combined using bitwise OR and saved successfully.")
    

if __name__=='__main__':

    datasets =["data_exp"]
    video_names = ["IMG_1779.MOV"]
    # os.listdir("rawvideo")
    for dataset in datasets:
        for video_name in video_names:
            print("run romp data pipeline")
            current_dir = os.getcwd()
            video_path = f'{current_dir}/RTFVHP/datasets/rawvideo/{video_name}'
            data_folder = f'{current_dir}/RTFVHP/datasets/processed/{dataset}/{video_name[:-4]}'
            image_folder = f'{current_dir}/RTFVHP/datasets/processed/{dataset}/{video_name[:-4]}/images/{video_name[:-4]}'
            mask_folder = f'{current_dir}/RTFVHP/datasets/processed/{dataset}/{video_name[:-4]}/masks/{video_name[:-4]}'
            smpl_mask_folder = f'{current_dir}/RTFVHP/datasets/processed/{dataset}/{video_name[:-4]}/smpl_masks/{video_name[:-4]}'
            combine_mask_folder = f'{current_dir}/RTFVHP/datasets/processed/{dataset}/{video_name[:-4]}/combine_mask_folder/{video_name[:-4]}'
            extract_frames(video_path, image_folder,10)

            # Load the DeepLabV3 model for mask prediciton
            model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
            model.eval()
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)
            #extract mask
            for i in tqdm(os.listdir(image_folder)):
                image_path = f'{os.path.join(image_folder,i)}'
                mask_path = os.path.join(mask_folder, i[:-4] + '.png')
                create_mask(image_path, mask_path)

            # extract smpl_masks
            if not os.path.exists(smpl_mask_folder):
                os.makedirs(smpl_mask_folder)
            getSMPLmask(image_folder,smpl_mask_folder)

            # the mask combination
            combine_mask(mask_folder,smpl_mask_folder,combine_mask_folder)
            # use romp to estimate the camera
            image_path = f'{current_dir}/RTFVHP/datasets/processed/{dataset}/{video_name[:-4]}/images/{video_name[:-4]}/000000.jpg'
            imsize = cv2.imread(image_path).shape[:2][::-1]
            
            # #calculate according to fov
            fov = 60
            H = max(imsize)
            focal_length=H/2. * 1./np.tan(np.radians(fov/2))

            metadata=rompEstimation(image_folder,focal_length,imsize)
            with open(os.path.join(data_folder,'metadata.json'),'w') as outfile:
                json.dump(metadata,outfile)

            # process to the humannerf data style
            prepareData(data_folder)
            
            
