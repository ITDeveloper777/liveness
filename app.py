import os, cv2
import numpy as np
import math
import time, pickle
from src.liveness_detect import LivenessDetect
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.analyze_face.analysis import FaceAnalysis
import warnings

warnings.simplefilter("ignore")

def get_filepaths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths

faceapp = FaceAnalysis(name='analg_face', root='.')
faceapp.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
recognizer = pickle.loads(open("./face_db/embeddings.dat", "rb").read())
le = pickle.loads(open("./face_db/le.dat", "rb").read())

def get_similarity(emb1, emb2):
    dot = np.sum(np.multiply(emb1, emb2), axis=0)
    norm = np.linalg.norm(emb1, axis=0) * np.linalg.norm(emb2, axis=0)
    similarity = min(1, max(-1, dot / norm))
    cosdist = min(0.5, np.arccos(similarity) / math.pi)
    pcnt = 0
    thr = 0.35
    if cosdist <= thr:
        pcnt = (0.2/thr) * cosdist
    elif cosdist > thr and cosdist <= 0.5:
        pcnt = 5.33333 * cosdist - 1.66667
    pcnt = (1.0 - pcnt) * 100
    pcnt = min(100, pcnt)
    return pcnt

def match(vector):
    encoding = vector
    max_sim = 0
    for i in range(len(recognizer["embeddings"])):
        db_enc = recognizer["embeddings"][i]
        name = recognizer["names"][i]
        sim = get_similarity(encoding, db_enc)

        if sim > max_sim:
            max_sim = sim
            identity = name
    if max_sim < 75:
        identity = "NONE"
    return max_sim, identity

def identify_face(image):
    faces = faceapp.get(image)
    largest_face = None
    if len(faces) > 0:
        max_box = None
        for face in faces:
            box = face.bbox.astype(np.int32)
            area =  (box[3] - box[1]) * (box[2]-box[0])
            
            max_area = 0
            try:
                max_area = (max_box[3]-max_box[1])*(max_box[2]-max_box[0])
            except:
                max_area = 0
            if area > max_area:
                 max_box = box
                 largest_face = face
        similarity, name = match(largest_face.embedding)
        return similarity, name
    else:
        return 0, "NONE"

def test(data_dir, model_dir):
    model_test = dict()
    for model_name in os.listdir(model_dir):
        model_test[model_name] = LivenessDetect(os.path.join(model_dir, model_name))

    image_cropper = CropImage()

    full_file_paths = get_filepaths(data_dir)
    total_count = 0
    real_count = 0
    lowest_real_val = 1
    index = 0
    for fpath in full_file_paths:
        if not os.path.isdir(fpath) and (fpath.lower().endswith('.jpg') or fpath.lower().endswith('.png')):
            image = cv2.imread(fpath)
            prediction = np.zeros((1, 3))
            test_speed = 0
            num_models = 0
            start = time.time()
            similarity, name = identify_face(image)

            for model_name in os.listdir(model_dir):
                image_bbox = model_test[model_name].get_bbox(image)
                h_input, w_input, _, scale = parse_model_name(model_name)
                param = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False
                img = image_cropper.crop(**param)
                
                predict_val = model_test[model_name].eval(img)
                prediction += predict_val
                num_models += 1

        test_speed = time.time() - start
        prediction = prediction / num_models
        total_count = total_count + 1
        if prediction[0][1] > 0.5:
            real_count = real_count + 1
            print(fpath + "------ {:.2f}s".format(test_speed) + " --- {:.3f}".format(prediction[0][1]) + " : " + name + "(" + str(similarity) +") ----- Real")
        else:
             print(fpath + "------ {:.2f}s".format(test_speed) + " --- {:.3f}".format(prediction[0][1]) + " : " + name + "(" + str(similarity) + ") ----- Fake")
        if lowest_real_val > prediction[0][1]:
            lowest_real_val = prediction[0][1]
    print('total : ' + str(total_count) + " --- As real :" + str(real_count) + " --- rate : " + str(float(real_count) / total_count))
    print("LOWEST value:" + str(lowest_real_val))

if __name__ == "__main__":
    test("./real_and_spoof/", "./models/liveness")
    # test("./zhong/", "./models/liveness")
    # test("./dataset/real/", "./models/liveness")
