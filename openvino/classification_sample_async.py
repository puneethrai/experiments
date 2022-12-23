#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log
import sys
import time

import cv2
import numpy as np
from openvino.preprocess import PrePostProcessor
from openvino.runtime import AsyncInferQueue, Core, InferRequest, Layout, Type

# --------------------------- Step A. Import Path to get the location of performance.txt -------------------------------
from pathlib import Path

# --------------------------- Step B. Declare Required Global variables ------------------------------------------------

start_time = 0 # Time when inference started
images_len = 0 # Number of images used
counter = 0 # Counter to measure number of inference completed
load_start_time = 0 # Start time to measure when model is loaded onto CPU
load_end_time = 0 # End time when model loaded onto CPU is completed

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action='help',
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model', type=str, required=True,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-i', '--input', type=str, required=True, nargs='+',
                      help='Required. Path to an image file(s).')
    args.add_argument('-d', '--device', type=str, default='CPU',
                      help='Optional. Specify the target device to infer on; CPU, GPU, MYRIAD, HDDL or HETERO: '
                      'is acceptable. The sample will look for a suitable plugin for device specified. '
                      'Default value is CPU.')
    args.add_argument('-p', '--performance', type=str, required=True,
                      help='Required. Path to write the performance output data')
    # fmt: on
    return parser.parse_args()


def completion_callback(infer_request: InferRequest, image_path: str) -> None:
    predictions = next(iter(infer_request.results.values()))

    # Change a shape of a numpy.ndarray with results to get another one with one dimension
    probs = predictions.reshape(-1)

    # Get an array of 10 class IDs in descending order of probability
    top_10 = np.argsort(probs)[-10:][::-1]

    header = 'class_id probability'

    log.info(f'Image path: {image_path}')
    log.info('Top 10 results: ')
    log.info(header)
    log.info('-' * len(header))

    for class_id in top_10:
        probability_indent = ' ' * (len('class_id') - len(str(class_id)) + 1)
        log.info(f'{class_id}{probability_indent}{probs[class_id]:.7f}')

# --------------------------- Step F. When inference of the image is completed then we measure its performance ------------------------------------------------
    global counter, images_len
    counter += 1
    if counter >= images_len:
        print_performance_report()

    log.info('')

# --------------------------- Step G. When inference of the image is completed then we measure its performance ------------------------------------------------
def print_performance_report():
    global start_time, images_len
    end_time = time.perf_counter()
    duration = end_time - start_time
    fps = images_len / duration
    args = parse_args()
    # Recommended to place performance.txt file to be placed inside FP32 (which ever precision has been used) folder
    output_path = f'{args.performance}/FP32/'
    log.info(f"Loaded model to {args.device} in {load_end_time-load_start_time:.2f} seconds.")
    log.info(f"Total time for {counter} frames: {duration:.2f} seconds, fps:{fps:.2f}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    f = open(f'{output_path}/performance.txt', 'w')
    f.write(f'Throughput: {fps:.2f} FPS\nLatency: {duration:.2f} s')
    f.close()

def main() -> int:
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()

# --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = Core()

# --------------------------- Step 2. Read a model --------------------------------------------------------------------
    log.info(f'Reading the model: {args.model}')
    # (.xml and .bin files) or (.onnx file)
    model = core.read_model(args.model)

    if len(model.inputs) != 1:
        log.error('Sample supports only single input topologies')
        return -1

    if len(model.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1

# --------------------------- Step 3. Set up input --------------------------------------------------------------------
    # Read input images
    images = [cv2.imread(image_path) for image_path in args.input]
# --------------------------- Step C. Initialize the images length ----------------------------------------------------
    global images_len
    images_len = len(images)

    # Resize images to model input dims
    _, _, h, w = model.input().shape
    resized_images = [cv2.resize(image, (w, h)) for image in images]

    # Add N dimension
    input_tensors = [np.expand_dims(image, 0) for image in resized_images]

# --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
    ppp = PrePostProcessor(model)

    # 1) Set input tensor information:
    # - input() provides information about a single model input
    # - precision of tensor is supposed to be 'u8'
    # - layout of data is 'NHWC'
    ppp.input().tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC'))  # noqa: N400

    # 2) Here we suppose model has 'NCHW' layout for input
    ppp.input().model().set_layout(Layout('NCHW'))

    # 3) Set output tensor information:
    # - precision of tensor is supposed to be 'f32'
    ppp.output().tensor().set_element_type(Type.f32)

    # 4) Apply preprocessing modifing the original 'model'
    model = ppp.build()

# --------------------------- Step 5. Loading model to the device -----------------------------------------------------
    global load_start_time, load_end_time, start_time

# --------------------------- Step D. To measure model load time ------------------------------------------------------
    load_start_time = time.perf_counter()
    log.info('Loading the model to the plugin')
    compiled_model = core.compile_model(model, args.device)
    load_end_time = time.perf_counter()

# --------------------------- Step E. Initialize the start time of inference ------------------------------------------
    start_time = time.perf_counter()

# --------------------------- Step 6. Create infer request queue ------------------------------------------------------
    log.info('Starting inference in asynchronous mode')
    # create async queue with optimal number of infer requests
    infer_queue = AsyncInferQueue(compiled_model)
    infer_queue.set_callback(completion_callback)

# --------------------------- Step 7. Do inference --------------------------------------------------------------------
    for i, input_tensor in enumerate(input_tensors):
        infer_queue.start_async({0: input_tensor}, args.input[i])

    infer_queue.wait_all()
# ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
