import pytest

import cc3d
import numpy as np

TEST_TYPES = [
  np.int8, np.int16, np.int32, np.int64,
  np.uint8, np.uint16, np.uint32, np.uint64,
]

def test_2d_square():
  for dtype in TEST_TYPES:
    input_labels = np.zeros( (16,16,1), dtype=dtype )
    input_labels[:8,:8,:] = 8
    input_labels[8:,:8,:] = 9
    input_labels[:8,8:,:] = 10
    input_labels[8:,8:,:] = 11

    output_labels = cc3d.connected_components(input_labels).astype(dtype)
    output_labels = output_labels[:,:,0]

    ground_truth = np.array([
      [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
      [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
    ], dtype=dtype)

    assert np.all(output_labels == ground_truth)

def test_2d_cross():
  for dtype in TEST_TYPES:
    input_labels = np.zeros( (17,17,1), dtype=dtype )
    input_labels[:] = 1
    input_labels[:,8,:] = 0
    input_labels[8,:,:] = 0

    output_labels = cc3d.connected_components(input_labels).astype(dtype)
    output_labels = output_labels[:,:,0]

    ground_truth = np.array([
      [1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4],
    ], dtype=dtype)

    assert np.all(output_labels == ground_truth)

    input_labels[9:,9:,:] = 2
    output_labels = cc3d.connected_components(input_labels).astype(dtype)
    output_labels = output_labels[:,:,0]
    assert np.all(output_labels == ground_truth)

def test_2d_cross_with_intruder():
  input_labels = np.zeros( (5,5,1), dtype=np.uint8, )
  input_labels[:] = 1
  input_labels[:,2,:] = 0
  input_labels[2,:,:] = 0
  input_labels[3:,3:,:] = 2
  input_labels[3,3,:] = 1

  output_labels = cc3d.connected_components(input_labels).astype(np.uint8)
  output_labels = output_labels[:,:,0]

  ground_truth = np.array([
    [1, 1, 0, 2, 2],
    [1, 1, 0, 2, 2],
    [0, 0, 0, 0, 0],
    [3, 3, 0, 4, 5],
    [3, 3, 0, 5, 5],
  ], dtype=np.uint8)

  assert np.all(output_labels == ground_truth)

def test_3d_all_different():
  input_labels = np.arange(0, 100 ** 3).astype(np.uint32)
  input_labels = input_labels.reshape((100,100,100))

  output_labels = cc3d.connected_components(input_labels)

  assert np.unique(output_labels).shape[0] == 100 ** 3


def test_3d_cross():
  for dtype in TEST_TYPES:
    print(dtype)
    input_labels = np.zeros( (7,7,7), dtype=dtype )
    input_labels[:] = 1
    input_labels[:,3,:] = 0
    input_labels[:,:,3] = 0

    output_labels = cc3d.connected_components(input_labels).astype(dtype)

    ground_truth = np.array([
     [[1, 1, 1, 0, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 0, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4]],

     [[1, 1, 1, 0, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 0, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4]],

     [[1, 1, 1, 0, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 0, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4]],

     [[1, 1, 1, 0, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 0, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4]],

     [[1, 1, 1, 0, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 0, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4]],

     [[1, 1, 1, 0, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 0, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4]],

     [[1, 1, 1, 0, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 0, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4]]], dtype=dtype)

    assert np.all(output_labels == ground_truth)

    input_labels[:,4:,4:] = 2
    output_labels = cc3d.connected_components(input_labels).astype(dtype)
    assert np.all(output_labels == ground_truth)


def test_512_cube_no_segfault_no_jitsu():
  input_labels = np.arange(0, 512 ** 3).astype(np.uint64).reshape((512,512,512))
  output_labels = cc3d.connected_components(input_labels)
