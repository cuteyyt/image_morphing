import argparse
import os
import shutil
from pathlib import Path
from time import time

import cv2
import dlib
import numpy as np
from scipy.ndimage import map_coordinates
from tqdm import tqdm

landmark_dict = {
    "left_eye": np.arange(36, 42),
    "left_eyebrow": np.arange(17, 22),
    "right_eye": np.arange(42, 48),
    "right_eyebrow": np.arange(22, 27),
    "nose": np.arange(31, 36),
    "nose_bridge": np.arange(27, 31),
    "lips_inner": np.arange(60, 68),
    "lips_outer": np.arange(48, 60),
    "face": np.arange(0, 17),
}


def cal_perpendicular_vector(vec):
    vec_length = np.sqrt(np.sum(vec ** 2, axis=1, keepdims=True))
    homo_vec = np.pad(vec, ((0, 0), (0, 1)), mode="constant")  # pad to R3, pad zeros
    z_axis = np.zeros(homo_vec.shape)
    z_axis[:, -1] = 1

    per_vec = np.cross(homo_vec, z_axis)
    per_vec = per_vec[:, :-1]  # ignore z axis
    per_length = np.sqrt(np.sum(per_vec ** 2, axis=1, keepdims=True))
    per_vec = per_vec / (per_length + 1e-8)  # now sum = 1
    per_vec *= vec_length
    return per_vec


def mapping(img, p1, q1, p_inter, q_inter, p=0.5, a=1, b=2, eps=1e-8):
    src_line_vec = q1 - p1
    src_per_vec = cal_perpendicular_vector(src_line_vec)
    dst_line_vec = q_inter - p_inter
    dst_per_vec = cal_perpendicular_vector(dst_line_vec)

    image_size = img.shape[0]
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))

    x_d = np.dstack([x, y])
    x_d = x_d.reshape((-1, 1, 2))

    to_p_vec = x_d - p_inter
    to_q_vec = x_d - q_inter
    u = np.sum(to_p_vec * dst_line_vec, axis=-1) / (np.sum(dst_line_vec ** 2, axis=1) + eps)
    v = np.sum(to_p_vec * dst_per_vec, axis=-1) / (np.sqrt(np.sum(dst_line_vec ** 2, axis=1)) + eps)

    x_s = \
        np.expand_dims(p1, 0) + np.expand_dims(u, -1) * np.expand_dims(src_line_vec, 0) + \
        np.expand_dims(v, -1) * np.expand_dims(src_per_vec, 0) / \
        (np.sqrt(np.sum(src_line_vec ** 2, axis=1)).reshape(1, -1, 1) + eps)
    d = x_s - x_d
    to_p_mask = (u < 0).astype(np.float64)
    to_q_mask = (u > 1).astype(np.float64)
    to_line_mask = np.ones(to_p_mask.shape) - to_p_mask - to_q_mask

    to_p_dist = np.sqrt(np.sum(to_p_vec ** 2, axis=-1))
    to_q_dist = np.sqrt(np.sum(to_q_vec ** 2, axis=-1))
    to_line_dist = np.abs(v)
    dist = to_p_dist * to_p_mask + to_q_dist * to_q_mask + to_line_dist * to_line_mask
    dest_line_length = np.sqrt(np.sum(dst_line_vec ** 2, axis=-1))
    weight = (dest_line_length ** p) / ((a + dist) ** b + eps)
    weighted_d = np.sum(d * np.expand_dims(weight, -1), axis=1) / (np.sum(weight, -1, keepdims=True) + eps)

    x_d = x_d.squeeze()
    x_s = x_d + weighted_d
    x_s_ij = x_s[:, ::-1]

    warped = np.zeros((image_size * image_size, img.shape[2]))
    for i in range(img.shape[2]):
        warped[:, i] = map_coordinates(img[:, :, i], x_s_ij.T, mode="nearest")
    warped = warped.reshape((image_size, image_size, -1)).squeeze()
    return warped.astype(np.uint8)


def get_intermediate_lines(p1, q1, p2, q2, alpha=0.5):
    p = p1 * alpha + p2 * (1 - alpha)
    q = q1 * alpha + q2 * (1 - alpha)
    return p, q


def merge(img1, p1, q1, img2, p2, q2, alpha=0.5, p=0.5, a=1, b=2, eps=1e-8):
    p_inter, q_inter = get_intermediate_lines(p1, q1, p2, q2, alpha)
    warped_1 = mapping(img1, p1, q1, p_inter, q_inter, p, a, b, eps)
    warped_2 = mapping(img2, p2, q2, p_inter, q_inter, p, a, b, eps)
    merged = warped_1 * alpha + warped_2 * (1 - alpha)
    return merged.astype(np.uint8)


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def draw_face(img, rect, lines_start, lines_end, landmarks, path):
    # draw rect
    (x, y, w, h) = rect_to_bb(rect)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # b, g, r

    # draw landmark points
    for x, y in landmarks:
        r = 2
        cv2.circle(img, (x, y), r, (0, 0, 255), -1)
    # draw feature lines
    for (x1, y1), (x2, y2) in zip(lines_start, lines_end):
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), cv2.LINE_AA)
    cv2.imwrite(path, img)


def draw(img, detector, predictor, path):
    line_start, line_end, landmarks, rect = get_line_start_and_end(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), detector,
                                                                   predictor)
    draw_face(img, rect, line_start, line_end, landmarks, path)


def parsing_face(img, detector, predictor):
    try:
        rect = detector(img, 1)[0]
    except Exception as e:
        raise Exception(f'img has no faces {e}')
    landmarks = predictor(img, rect)

    coords = np.zeros((landmarks.num_parts, 2), dtype=np.int32)
    for i in range(0, landmarks.num_parts):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    return rect, np.array(coords)


def get_line_start_and_end(gray_img, detector, predictor):
    rect, landmarks = parsing_face(gray_img, detector, predictor)

    landmarks_coordinates = list()
    for key, pts in landmark_dict.items():
        coords = np.stack([pts[:-1], pts[1:]]).T
        landmarks_coordinates.append(coords)
    landmarks_coordinates = np.concatenate(landmarks_coordinates)

    line_start = landmarks[landmarks_coordinates[:, 0]]
    line_end = landmarks[landmarks_coordinates[:, 1]]

    return line_start.astype(np.float64), line_end.astype(np.float64), landmarks, rect


def extract_face(path, detector, predictor, img_size=256):
    color_img = cv2.imread(path)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    height, width = gray_img.shape[0], gray_img.shape[1]

    rect, landmarks = parsing_face(gray_img, detector, predictor)

    padding = 10
    max_x, min_x = np.max(landmarks[:, 0]), np.min(landmarks[:, 0])
    max_y, min_y = np.max(landmarks[:, 1]), np.min(landmarks[:, 1])

    gray_cropped = gray_img[
                   max(0, min_y - padding):min(max_y + padding, height),
                   max(0, min_x - padding):min(max_x + padding, width)
                   ]
    color_cropped = color_img[
                    max(0, min_y - padding):min(max_y + padding, height),
                    max(0, min_x - padding):min(max_x + padding, width)
                    ]

    gray_resized = cv2.resize(gray_cropped, (img_size, img_size))
    color_resized = cv2.resize(color_cropped, (img_size, img_size))

    return gray_resized, color_resized, color_img


def morphing_two_figs(opt, src_img_path, dst_img_path):
    predictor_path, img_size, p, a, b, steps = opt.predictor_path, opt.img_size, opt.p, opt.a, opt.b, opt.steps

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    gray_img1, color_img1, ori_color1 = extract_face(src_img_path, detector, predictor, img_size)
    gray_img2, color_img2, ori_color2 = extract_face(dst_img_path, detector, predictor, img_size)
    p1, q1, landmarks1, _ = get_line_start_and_end(gray_img1, detector, predictor)
    p2, q2, landmarks2, _ = get_line_start_and_end(gray_img2, detector, predictor)

    parsing_src_img_path = f'{opt.inter_path}/{Path(src_img_path).resolve().name}'
    parsing_dst_img_path = f'{opt.inter_path}/{Path(dst_img_path).resolve().name}'
    if not os.path.exists(parsing_src_img_path):
        draw(ori_color1, detector, predictor, parsing_src_img_path)
    if not os.path.exists(parsing_dst_img_path):
        draw(ori_color2, detector, predictor, parsing_dst_img_path)

    for alpha in np.linspace(1., 0, steps):
        merged = merge(color_img1, p1, q1, color_img2, p2, q2, alpha, p, a, b, args.eps)
        cv2.imwrite(increment_path('materials'), merged)


def increment_path(path='materials'):
    file_list = sorted(os.listdir(path), key=lambda x: int(Path(x).resolve().stem))
    if len(file_list):
        filename = f'{int(Path(file_list[-1]).resolve().stem) + 1}.png'
    else:
        filename = '1.png'
    return f'{path}/{filename}'


def morphing_folder(opt):
    input_path = opt.input_path

    file_list = [f'{input_path}/{file}' for file in os.listdir(input_path)]
    for i in tqdm(sorted(range(1, len(file_list)))):
        filepath1 = file_list[i - 1]
        filepath2 = file_list[i]
        morphing_two_figs(opt, filepath1, filepath2)


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default='imgs/test')
    parser.add_argument("--src_img_path", type=str, default='imgs/test/000001.jpg')
    parser.add_argument("--dst_img_path", type=str, default='imgs/test/000002.jpg')
    parser.add_argument("--output_path", type=str, default='results/morphing.mp4')
    parser.add_argument("--predictor_path", type=str, default="ckpt/shape_predictor_68_face_landmarks.dat")

    parser.add_argument("--img_size", type=int, help="size of cropped and resized face", default=256)
    parser.add_argument("--a", type=float, help="parameter (a) in paper", default=1)
    parser.add_argument("--b", type=float, help="parameter (b) in paper", default=2)
    parser.add_argument("--p", type=float, help="parameter (p) in paper", default=0.5)
    parser.add_argument('--eps', type=float, default=1e-8)

    parser.add_argument("--frames", type=int, help="frames in morphing video)", default=30)
    parser.add_argument("--frames_per_pair", type=int, help="frames per pair", default=1)

    opt = parser.parse_args()

    assert os.path.exists(opt.input_path) or (os.path.exists(opt.src_img_path) and os.path.exists(
        opt.dst_img_path)), f'either --input_path or (--src_img_path and --dst_img_path) should be valid)'

    opt.inter_path = f'{Path(opt.output_path).parent}/inter'
    os.makedirs(opt.inter_path, exist_ok=True)

    assert Path(opt.output_path).resolve().suffix == '.mp4', '--output_path must be mp4 format'

    opt.steps = opt.frames * opt.frames_per_pair

    return opt


if __name__ == "__main__":
    args = parse_opt()

    if os.path.exists(args.input_path):
        mode = 'folder'
    else:
        mode = 'picture'

    if os.path.exists('materials'):
        shutil.rmtree('materials')
    os.makedirs('materials', exist_ok=True)
    start = time()
    if mode == 'folder':
        morphing_folder(args)
    else:
        morphing_two_figs(args, args.src_img_path, args.dst_img_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(args.output_path, fourcc, args.frames, (args.img_size, args.img_size))
    for item in tqdm(sorted(os.listdir('materials'), key=lambda x: int(Path(x).resolve().stem))):
        image = cv2.imread(f'materials/{item}')
        # cv2.imshow('debug', image)
        # cv2.waitKey(0)
        video.write(image)
    shutil.rmtree('materials')
    end = time()
    print('Spent: {:.3f} seconds'.format(end - start))
