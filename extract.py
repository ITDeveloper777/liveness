from sklearn.preprocessing import LabelEncoder
from imutils import paths
import numpy as np
import pickle
import cv2
import os
from src.analyze_face.analysis import FaceAnalysis

dataset_path = "./faces"
embeddings_path = "./face_db/embeddings.dat"
le_path = "./face_db/le.dat"
imagePaths = list(paths.list_images(dataset_path))
knownEmbeddings = []
knownNames = []

faceapp = FaceAnalysis(name='analg_face', root='.')
faceapp.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

for (i, imagePath) in enumerate(imagePaths):
	name = imagePath.split(os.path.sep)[-1]
	image = cv2.imread(imagePath)
	faces = faceapp.get(image)
	largest_face = None
	if len(faces) > 0:
		max_box = None
		for face in faces:
			box = face.bbox.astype(np.int)
			area =  (box[3] - box[1]) * (box[2]-box[0])
			if max_box == None or area > (max_box[3]-max_box[1])*(max_box[2]-max_box[0]):
				max_box = box
				largest_face = face
	face_image = image[max_box[1]: max_box[3], max_box[0]:max_box[2]]
	filename, ext = os. path. splitext(name)
	knownNames.append(filename)
	knownEmbeddings.append(largest_face.embedding.flatten())

data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(embeddings_path, "wb")
f.write(pickle.dumps(data))
f.close()

le = LabelEncoder()
labels = le.fit_transform(data["names"])
f = open(le_path, "wb")
f.write(pickle.dumps(le))
f.close()
print(knownNames)