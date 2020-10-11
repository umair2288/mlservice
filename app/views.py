from flask import jsonify , request
from app import app
import os
import json
from .inference import LeavesSegmentation, InferenceConfig
import skimage


@app.route("/predict" , methods=["POST"])
def hello():
    config = InferenceConfig()
    img_url = request.json.get("img",None)
    if img_url :
        IMAGE_PATH = img_url #os.path.join("app","model","mask_rcnn_leave_0015.h5")
        MODEL_PATH =  os.path.join("app","model","mask_rcnn_leave_0015.h5") #"E:\\Lakna\\Flask\\mlservice\\app\\model\\mask_rcnn_leave_0015.h5"
        print(os.path.join("app","model","mask_rcnn_leave_0015.h5"))
        lmodel = LeavesSegmentation(config, path=MODEL_PATH, logdir= os.path.join("model","logs"))
        image = skimage.io.imread(IMAGE_PATH)[:,:,:3]# 
        # predict
        r, im = lmodel.predict(image)
        # print(r)
        resonse  = {
            "success":True,
            "message": "prediction sucessfull",
            "data" : {
                "healthy" : r
            }
        }
        return jsonify(resonse)
    else:
        return jsonify({"success": False , "message":"Image url not provided"})
    