from typing import Dict, List, Union, Tuple

import numpy as np
import onnxruntime as ort

from walrus_yolo.utils import non_max_suppression, letterbox, scale_coords


class ObjectDetector:
    def __init__(
        self,
        model_path: str,
        image_size: Tuple[int, int] = (1280, 1280),
        device: str = "gpu",
        device_id: int = 0,
        threads: int = 1,
        log_level: int = 3,
    ) -> None:
        self._image_size = image_size
        self._conf_threshold = 0.5
        self._nms_threshold = 0.4
        self._yolo_path = model_path

        ort_device = ort.get_device().lower()
        device = device.lower()
        if device == "cpu":
            self._providers = ["CPUExecutionProvider"]
            self._provider_options = [{}]
        elif device == "gpu" and ort_device == "gpu":
            self._providers = ["CUDAExecutionProvider"]
            self._provider_options = [{"device_id": str(device_id)}]
        else:
            raise ValueError(f"ONNXRuntime does not support '{device}' device")
        self._onnx_options = ort.SessionOptions()
        self._onnx_options.log_severity_level = log_level
        if threads > 0:
            self._onnx_options.inter_op_num_threads = threads
            self._onnx_options.intra_op_num_threads = threads
        else:
            raise ValueError(f"Incorrect num of threads: {threads}")
        self._yolo_model = ort.InferenceSession(
            self._yolo_path,
            providers=self._providers,
            sess_options=self._onnx_options,
            provider_options=self._provider_options,
        )
        self._input_name = self._yolo_model.get_inputs()[0].name

    def __call__(
        self, image: np.ndarray
    ) -> List[Dict["str", Union[List[int], float, str]]]:

        orig_shape = image.shape
        image_flip = image.copy()
        image_flip = image_flip[:, ::-1, :]

        image = self._preprocessing(image)
        image_flip = self._preprocessing(image_flip)
        batch_images = np.concatenate((image, image_flip), axis=0)
        output = np.asarray(self._get_model_output(batch_images))
        output[1, :, 0] = self._image_size[0] - output[1, :, 0]
        output = output.reshape((1, -1, output.shape[2]))

        # orig_shape = image.shape
        # quarter_image = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5)
        # im_list_2d = [
        #     [quarter_image, quarter_image[:, ::-1, :]],
        #     [np.where((quarter_image * 1.2) > 255, 255,
        #               (quarter_image * 1.2)).astype(np.uint8),
        #      np.array(quarter_image * 0.8, dtype=np.uint8)]
        # ]
        # concat_image = cv2.vconcat(
        #     [cv2.hconcat(im_list_h) for im_list_h in im_list_2d]
        # )
        #
        # if image.shape != concat_image.shape:
        #     concat_image = cv2.copyMakeBorder(
        #         concat_image, top=0, left=0,
        #         bottom=(image.shape[0] - concat_image.shape[0]),
        #         right=(image.shape[1] - concat_image.shape[1]),
        #         borderType=cv2.BORDER_CONSTANT, value=0
        #     )
        #
        # image = self._preprocessing(image)
        # concat_image = self._preprocessing(concat_image)
        #
        # batch_images = np.concatenate((image, concat_image), axis=0)
        # _, _, im_height, im_width = batch_images.shape
        # output = np.asarray(self._get_model_output(batch_images))
        #
        # first_quarter_mask = ((output[1, :, 0] > (im_width // 2)) &
        #                       (output[1, :, 1] <= (im_height // 2)))
        # first_quarter_mask = np.vstack([first_quarter_mask
        #                                 for _ in range(4)]).T
        #
        # second_quarter_mask = ((output[1, :, 0] <= (im_width // 2)) &
        #                        (output[1, :, 1] <= (im_height // 2)))
        # second_quarter_mask = np.vstack([second_quarter_mask
        #                                  for _ in range(4)]).T
        #
        # third_quarter_mask = ((output[1, :, 0] <= (im_width // 2)) &
        #                       (output[1, :, 1] > (im_height // 2)))
        # third_quarter_mask = np.vstack([third_quarter_mask
        #                                 for _ in range(4)]).T
        #
        # fourth_quarter_mask = ((output[1, :, 0] > (im_width // 2)) &
        #                        (output[1, :, 1] > (im_height // 2)))
        # fourth_quarter_mask = np.vstack([fourth_quarter_mask
        #                                  for _ in range(4)]).T
        #
        # # Upscale first quarter of TTA image
        # output[1, :, 0] = np.where(
        #     first_quarter_mask[:, 0],
        #     im_width - (output[1, :, 0] - im_width // 2) * 2, output[1, :, 0]
        # )
        # output[1, :, 1:4] = np.where(
        #     first_quarter_mask[:, 1:4], output[1, :, 1:4] * 2,
        #     output[1, :, 1:4]
        # )
        #
        # # Upscale second quarter of TTA image
        # output[1, :, :4] = np.where(
        #     second_quarter_mask, output[1, :, :4] * 2, output[1, :, :4]
        # )
        #
        # # Upscale third quarter of TTA image
        # output[1, :, 1] = np.where(
        #     third_quarter_mask[:, 1], output[1, :, 1] - im_height // 2,
        #     output[1, :, 1]
        # )
        # output[1, :, :4] = np.where(
        #     third_quarter_mask, output[1, :, :4] * 2, output[1, :, :4]
        # )
        #
        # # Upscale fourth quarter of TTA image
        # output[1, :, :2] = np.where(
        #     fourth_quarter_mask[:, :2],
        #     output[1, :, :2] - np.array([im_width // 2, im_height // 2]),
        #     output[1, :, :2]
        # )
        # output[1, :, :4] = np.where(
        #     fourth_quarter_mask, output[1, :, :4] * 2, output[1, :, :4]
        # )
        #
        # output = output.reshape((1, -1, output.shape[2]))

        output = non_max_suppression(
            output,
            conf_thres=self._conf_threshold,
            iou_thres=self._nms_threshold,
            agnostic=False,
        )

        detections = output[0]
        result = list()

        if len(detections):
            detections[:, :4] = scale_coords(
                image.shape[2:], detections[:, :4], orig_shape
            ).round()

            for detection in detections:
                class_id = int(detection[5])
                confidence = float(detection[4])
                x = int(detection[0])
                y = int(detection[1])
                w = int(detection[2] - x)
                h = int(detection[3] - y)

                result.append(
                    {
                        "class": class_id,
                        "box": [x, y, w, h],
                        "conf": round(confidence, 4),
                    }
                )

        return result

    def _get_model_output(self, img_in: np.ndarray) -> List[np.ndarray]:
        return self._yolo_model.run(None, {self._input_name: img_in})[0]

    def _preprocessing(self, image: np.ndarray) -> np.ndarray:
        resized = letterbox(image, new_shape=self._image_size, auto=True)[0]
        img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        return img_in
