import numpy as np
import box_utils_numpy as box_utils
import cv2



class UltraPredictor():
    def __init__(self, ort_session, threshold, class_names):
        self.ort_session = ort_session
        self.input_name = ort_session.get_inputs()[0].name
        self.threshold = threshold
        self.class_names = class_names
    
    def predict(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        confidences, boxes = self.ort_session.run(None, {self.input_name: image})
        boxes, labels, probs = self.in_predict(frame.shape[1], frame.shape[0], confidences, boxes, self.threshold)
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            print (box)
            label = f"{self.class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    
    def in_predict(self, width, height, confidences, boxes, prob_threshold=0.7, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                           iou_threshold=iou_threshold,
                                           top_k=top_k,
                                           )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]
