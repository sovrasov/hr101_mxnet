# coding: utf-8
from tiny_fd import TinyFacesDetector
import sys, os
import cv2 as cv
import argparse
import json

def main(args):
    detector = TinyFacesDetector(model_root=args.detector_root, prob_thresh=args.fd_t, gpu_idx=args.device, cudnn_autotune=False)
    cap = cv.VideoCapture(args.video)

    output_data = []
    video_basename = os.path.basename(args.video)

    cv.namedWindow('Tiny FD', cv.WINDOW_NORMAL)
    frame_id = 0
    while cv.waitKey(1) != ord('q'):
        has_frame, frame = cap.read()
        if not has_frame:
            break
        boxes = detector.detect(frame)
        print('Detected {} faces'.format(boxes.shape[0]))

        frame_data = dict()
        frame_data['id'] = frame_id
        frame_data['video'] = video_basename
        frame_data['objects'] = list()

        h, w = frame.shape[:2]
        for r in boxes:
            frame_data['objects'].append([float(r[0])/w, float(r[1])/h, float(r[2])/w, float(r[3])/h])
            cv.rectangle(frame, (r[0],r[1]), (r[2],r[3]), (255,255,0), 3)

        output_data.append(frame_data)
        cv.imshow('Tiny FD', frame)
        frame_id += 1

    json.dump(output_data, open(video_basename.split('.')[0] + '.json', 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('video', type=str, default=None, help='Input video.')
    parser.add_argument('--detector_root', type=str, default='./')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fd_t', type=float, default=0.5)
    main(parser.parse_args())
