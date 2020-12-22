import pytest
import sys

import cc3d
import numpy as np

TEST_TYPES = [
  np.int8, np.int16, np.int32, np.int64,
  np.uint8, np.uint16, np.uint32, np.uint64,
]

OUT_TYPES = [ np.uint16, np.uint32, np.uint64 ]

def gt_c2f(gt):
  f_gt = np.copy(gt)
  mx = np.max(gt) + 1
  f_gt[ f_gt == 2 ] = mx
  f_gt[ f_gt == 3 ] = 2
  f_gt[ f_gt == mx ] = 3
  return f_gt

@pytest.mark.parametrize("connectivity", (4, 6, 8, 18, 26))
@pytest.mark.parametrize("dtype", TEST_TYPES)
def test_2d_square(dtype, connectivity):
  def test(order, ground_truth):
    input_labels = np.zeros( (16,16), dtype=dtype, order=order )
    input_labels[:8,:8] = 8
    input_labels[8:,:8] = 9
    input_labels[:8,8:] = 10
    input_labels[8:,8:] = 11

    output_labels = cc3d.connected_components(
      input_labels, connectivity=connectivity
    ).astype(dtype)
    
    print(output_labels)

    assert np.all(output_labels == ground_truth.astype(dtype))

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
  ])

  test('C', ground_truth)
  test('F', ground_truth.T)

@pytest.mark.parametrize("connectivity", (4, 6, 8, 18, 26))
@pytest.mark.parametrize("dtype", TEST_TYPES)
def test_2d_rectangle(dtype, connectivity):
  def test(order, ground_truth):
    input_labels = np.zeros( (16,13), dtype=dtype, order=order )
    input_labels[:8,:8] = 8
    input_labels[8:,:8] = 9
    input_labels[:8,8:] = 10
    input_labels[8:,8:] = 11

    output_labels = cc3d.connected_components(
      input_labels, connectivity=connectivity
    ).astype(dtype)
    print(output_labels.shape)
    output_labels = output_labels[:,:]

    print(output_labels)

    assert np.all(output_labels == ground_truth.astype(dtype))

  ground_truth = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
    [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
    [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
    [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
    [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
    [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
    [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
    [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
  ])

  test('C', ground_truth)
  test('F', gt_c2f(ground_truth))

@pytest.mark.parametrize("connectivity", (4, 6, 8, 18, 26))
@pytest.mark.parametrize("dtype", TEST_TYPES)
def test_2d_cross(dtype, connectivity):
  def test(order, ground_truth):
    input_labels = np.zeros( (17,17), dtype=dtype, order=order)
    input_labels[:] = 1
    input_labels[:,8] = 0
    input_labels[8,:] = 0

    output_labels = cc3d.connected_components(input_labels, connectivity=connectivity).astype(dtype)
    print(output_labels)

    assert np.all(output_labels == ground_truth)

    input_labels[9:,9:] = 2
    output_labels = cc3d.connected_components(input_labels, connectivity=connectivity).astype(dtype)
    output_labels = output_labels[:,:]
    assert np.all(output_labels == ground_truth)

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
  ], dtype=np.uint32)

  test('C', ground_truth)
  test('F', gt_c2f(ground_truth))

@pytest.mark.parametrize("connectivity", (8, 18, 26))
def test_2d_diagonals_8_connected(connectivity):
  input_labels = np.array([
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  ], dtype=np.uint32)

  ground_truth = np.array([
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 1, 0, 0, 3, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 1, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 2, 0, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 6, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 6, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
  ], dtype=np.uint32)

  output_labels = cc3d.connected_components(input_labels, connectivity=connectivity)
  print(output_labels)
  assert np.all(output_labels == ground_truth)

@pytest.mark.parametrize("connectivity", (4, 6))
def test_2d_diagonals_4_connected(connectivity):
  input_labels = np.array([
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  ], dtype=np.uint32)

  ground_truth_sauf = np.array([
    [0,  0,  1, 0,  2,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
    [3,  0,  0, 4,  0,  0,  5,  0,  6,  6,  0,  0, 0, 0, 0, 0, 0],
    [0,  7,  0, 0,  8,  0,  0,  0,  6,  6,  0,  0, 0, 0, 0, 0, 0],
    [9,  0, 10, 0,  0,  0, 11, 11,  0,  0, 12, 12, 0, 0, 0, 0, 0],
    [0,  0,  0, 0,  0,  0, 11, 11,  0,  0, 12, 12, 0, 0, 0, 0, 0],
    [0,  0, 13, 0, 14,  0,  0,  0,  0, 15,  0,  0, 0, 0, 0, 0, 0],
    [0, 16,  0, 0,  0,  17, 0,  0, 18,  0,  0,  0, 0, 0, 0, 0, 0],
  ], dtype=np.uint32)

  ground_truth_bbdt = np.array([
    [0,  0,  2, 0,  4,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0],
    [1,  0,  0, 3,  0,  0,  5,  0,  6,  6,  0,  0, 0, 0, 0, 0, 0],
    [0,  8,  0, 0, 10,  0,  0,  0,  6,  6,  0,  0, 0, 0, 0, 0, 0],
    [7,  0,  9, 0,  0,  0, 11, 11,  0,  0, 12, 12, 0, 0, 0, 0, 0],
    [0,  0,  0, 0,  0,  0, 11, 11,  0,  0, 12, 12, 0, 0, 0, 0, 0],
    [0,  0, 13, 0, 14,  0,  0,  0,  0, 15,  0,  0, 0, 0, 0, 0, 0],
    [0, 16,  0, 0,  0,  17, 0,  0, 18,  0,  0,  0, 0, 0, 0, 0, 0],
  ], dtype=np.uint32)

  output_labels = cc3d.connected_components(input_labels, connectivity=connectivity)
  print(output_labels)
  assert np.all(output_labels == ground_truth_sauf) or np.all(output_labels == ground_truth_bbdt)

@pytest.mark.parametrize("connectivity", (4, 6, 8, 18, 26))
def test_2d_cross_with_intruder(connectivity):
  def test(order, ground_truth):
    input_labels = np.zeros( (5,5), dtype=np.uint8, order=order)
    input_labels[:] = 1
    input_labels[:,2] = 0
    input_labels[2,:] = 0
    input_labels[3:,3:] = 2
    input_labels[3,3] = 1

    output_labels = cc3d.connected_components(input_labels, connectivity=connectivity).astype(np.uint8)
    assert np.all(output_labels == ground_truth)

  ground_truth = np.array([
    [1, 1, 0, 2, 2],
    [1, 1, 0, 2, 2],
    [0, 0, 0, 0, 0],
    [3, 3, 0, 4, 5],
    [3, 3, 0, 5, 5],
  ], dtype=np.uint8)

  test("C", ground_truth)
  test("F", gt_c2f(ground_truth))

@pytest.mark.parametrize("order", ('C', 'F'))
def test_3d_all_different(order):
  input_labels = np.arange(0, 100 * 99 * 98).astype(np.uint32)
  input_labels = input_labels.reshape((100,99,98), order=order)

  output_labels = cc3d.connected_components(input_labels)

  assert np.unique(output_labels).shape[0] == 100*99*98
  assert output_labels.shape == (100, 99, 98)

@pytest.mark.parametrize("dtype", TEST_TYPES)
def test_3d_cross(dtype):
  def test(order, ground_truth):
    print(order)
    input_labels = np.zeros( (7,7,7), dtype=dtype, order=order )
    input_labels[:] = 1
    input_labels[:,3,:] = 0
    input_labels[:,:,3] = 0

    output_labels = cc3d.connected_components(input_labels).astype(dtype)
    print(output_labels)
    assert np.all(output_labels == ground_truth)

    input_labels[:,4:,4:] = 2
    output_labels = cc3d.connected_components(input_labels).astype(dtype)
    assert np.all(output_labels == ground_truth)

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
      [3, 3, 3, 0, 4, 4, 4]]], dtype=np.uint32)

  test('C', ground_truth)
  test('F', gt_c2f(ground_truth))

@pytest.mark.parametrize("dtype", TEST_TYPES)
def test_3d_cross_asymmetrical(dtype):
  def test(order, ground_truth):
    print(order)
    input_labels = np.zeros( (7,7,8), dtype=dtype, order=order )
    input_labels[:] = 1
    input_labels[:,3,:] = 0
    input_labels[:,:,3] = 0

    output_labels = cc3d.connected_components(input_labels).astype(dtype)
    print(output_labels)
    assert np.all(output_labels == ground_truth)

    input_labels[:,4:,4:] = 2
    output_labels = cc3d.connected_components(input_labels).astype(dtype)
    assert np.all(output_labels == ground_truth)

  ground_truth = np.array([
     [[1, 1, 1, 0, 2, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 0, 4, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4, 4]],

     [[1, 1, 1, 0, 2, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 0, 4, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4, 4]],

     [[1, 1, 1, 0, 2, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 0, 4, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4, 4]],

     [[1, 1, 1, 0, 2, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 0, 4, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4, 4]],

     [[1, 1, 1, 0, 2, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 0, 4, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4, 4]],

     [[1, 1, 1, 0, 2, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 0, 4, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4, 4]],

     [[1, 1, 1, 0, 2, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2, 2],
      [1, 1, 1, 0, 2, 2, 2, 2],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [3, 3, 3, 0, 4, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4, 4],
      [3, 3, 3, 0, 4, 4, 4, 4]],

    ], dtype=np.uint32)

  test('C', ground_truth)
  test('F', gt_c2f(ground_truth))

@pytest.mark.skipif(platform='win32')
@pytest.mark.xfail(raises=MemoryError, reason="Some build tools don't have enough memory for this.")
def test_512_cube_no_segfault_no_jitsu(): 
  input_labels = np.arange(0, 512 ** 3).astype(np.uint64).reshape((512,512,512))
  output_labels = cc3d.connected_components(input_labels)

def test_max_labels_nonsensical():
  input_labels = np.arange(0, 64 ** 3).astype(np.uint64).reshape((64,64,64))
  real_labels = cc3d.connected_components(input_labels, max_labels=64*64*64)
  zero_labels = cc3d.connected_components(input_labels, max_labels=0)
  negative_labels = cc3d.connected_components(input_labels, max_labels=-50)

  assert np.all(real_labels == zero_labels)
  assert np.all(real_labels == negative_labels)

def test_compare_scipy_26():
  import scipy.ndimage.measurements

  sx, sy, sz = 128, 128, 128
  labels = np.random.randint(0,2, (sx,sy,sz), dtype=np.bool)

  structure = [
    [[1,1,1], [1,1,1], [1,1,1]],
    [[1,1,1], [1,1,1], [1,1,1]],
    [[1,1,1], [1,1,1], [1,1,1]]
  ]

  cc3d_labels, Ncc3d = cc3d.connected_components(labels, connectivity=26, return_N=True)
  scipy_labels, Nscipy = scipy.ndimage.measurements.label(labels, structure=structure)

  print(cc3d_labels)
  print(scipy_labels)

  assert Ncc3d == Nscipy
  assert np.all(cc3d_labels == scipy_labels)

def test_compare_scipy_18():
  import scipy.ndimage.measurements

  sx, sy, sz = 256, 256, 256
  labels = np.random.randint(0,2, (sx,sy,sz), dtype=np.bool)

  structure = [
    [[0,1,0], [1,1,1], [0,1,0]],
    [[1,1,1], [1,1,1], [1,1,1]],
    [[0,1,0], [1,1,1], [0,1,0]]
  ]

  cc3d_labels, Ncc3d = cc3d.connected_components(labels, connectivity=18, return_N=True)
  scipy_labels, Nscipy = scipy.ndimage.measurements.label(labels, structure=structure)

  print(cc3d_labels)
  print(scipy_labels)

  assert Ncc3d == Nscipy
  assert np.all(cc3d_labels == scipy_labels)

def test_compare_scipy_6():
  import scipy.ndimage.measurements

  sx, sy, sz = 256, 256, 256
  labels = np.random.randint(0,2, (sx,sy,sz), dtype=np.bool)

  cc3d_labels, Ncc3d = cc3d.connected_components(labels, connectivity=6, return_N=True)
  scipy_labels, Nscipy = scipy.ndimage.measurements.label(labels)

  print(cc3d_labels)
  print(scipy_labels)

  assert Ncc3d == Nscipy
  assert np.all(cc3d_labels == scipy_labels)

# @pytest.mark.skipif("sys.maxsize <= 2**33")
# @pytest.mark.xfail(raises=MemoryError, reason="Some build tools don't have enough memory for this.")
# def test_sixty_four_bit():
#   input_labels = np.ones((1626,1626,1626), dtype=np.uint8)
#   cc3d.connected_components(input_labels, max_labels=3)  

@pytest.mark.parametrize("size", (255,256))
@pytest.mark.parametrize("zeroth_pass", (True,False))
def test_stress_upper_bound_for_binary_6(size, zeroth_pass):
  labels = np.zeros((size,size,size), dtype=np.bool)
  for z in range(labels.shape[2]):
    for y in range(labels.shape[1]):
      off = (y + (z % 2)) % 2
      labels[off::2,y,z] = True

  out = cc3d.connected_components(labels, connectivity=6, zeroth_pass=zeroth_pass)
  assert np.max(out) + 1 <= (256**3) // 2 + 1

@pytest.mark.parametrize("size", (255,256))
@pytest.mark.parametrize("zeroth_pass", (True,False))
def test_stress_upper_bound_for_binary_8(size, zeroth_pass):
  labels = np.zeros((size,size), dtype=np.bool)
  labels[0::2,0::2] = True

  out = cc3d.connected_components(labels, connectivity=8, zeroth_pass=zeroth_pass)
  assert np.max(out) + 1 <= (256**2) // 4 + 1

  for _ in range(10):
    labels = np.random.randint(0,2, (256,256), dtype=np.bool)
    out = cc3d.connected_components(labels, connectivity=8)
    assert np.max(out) + 1 <= (256**2) // 4 + 1    

@pytest.mark.parametrize("size", (255,256))
@pytest.mark.parametrize("zeroth_pass", (True,False))
def test_stress_upper_bound_for_binary_18(size, zeroth_pass):
  labels = np.zeros((size,size,size), dtype=np.bool)
  labels[::2,::2,::2] = True
  labels[1::2,1::2,::2] = True

  out = cc3d.connected_components(labels, connectivity=18, zeroth_pass=zeroth_pass)
  assert np.max(out) + 1 <= (256**3) // 4 + 1

  for _ in range(10):
    labels = np.random.randint(0,2, (256,256,256), dtype=np.bool)
    out = cc3d.connected_components(labels, connectivity=18, zeroth_pass=zeroth_pass)
    assert np.max(out) + 1 <= (256**3) // 4 + 1

@pytest.mark.parametrize("size", (255,256))
@pytest.mark.parametrize("zeroth_pass", (True,False))
def test_stress_upper_bound_for_binary_26(size, zeroth_pass):
  labels = np.zeros((size,size,size), dtype=np.bool)
  labels[::2,::2,::2] = True

  out = cc3d.connected_components(labels, connectivity=26, zeroth_pass=zeroth_pass)
  assert np.max(out) + 1 <= (256**3) // 8 + 1

  for _ in range(10):
    labels = np.random.randint(0,2, (256,256,256), dtype=np.bool)
    out = cc3d.connected_components(labels, connectivity=26, zeroth_pass=zeroth_pass)
    assert np.max(out) + 1 <= (256**3) // 8 + 1

@pytest.mark.parametrize("connectivity", (8, 18, 26))
@pytest.mark.parametrize("dtype", TEST_TYPES)
@pytest.mark.parametrize("order", ("C", "F"))
def test_all_zeros_3d(connectivity, dtype, order):
  labels = np.zeros((128,128,128), dtype=dtype, order=order)
  out = cc3d.connected_components(labels)
  assert np.all(out == 0)

@pytest.mark.parametrize("connectivity", (8, 18, 26))
@pytest.mark.parametrize("dtype", TEST_TYPES)
@pytest.mark.parametrize("order", ("C", "F"))
@pytest.mark.parametrize("lbl", (1, 100, 7))
def test_all_single_foreground(connectivity, dtype, order, lbl):
  labels = np.zeros((64,64,64), dtype=dtype, order=order) + lbl
  out = cc3d.connected_components(labels)
  assert np.all(out == 1)

@pytest.mark.parametrize("dtype", OUT_TYPES)
@pytest.mark.parametrize("order", ("C", "F"))
@pytest.mark.parametrize("in_place", (True, False))
@pytest.mark.parametrize("dims", (1,2,3))
def test_each(dtype, order, in_place, dims):
  shape = [128] * dims
  labels = np.random.randint(0,3, shape, dtype=dtype)

  for label, img in cc3d.each(labels, binary=False, in_place=in_place):
    assert np.all(img == (label * (labels == label)))

  for label, img in cc3d.each(labels, binary=True, in_place=in_place):
    assert np.all(img == (labels == label))

def test_region_graph_26():
  labels = np.zeros( (10, 10, 10), dtype=np.uint32 )

  labels[5,5,5] = 1
  labels[6,6,6] = 2
  labels[4,4,6] = 3
  labels[4,6,6] = 4
  labels[6,4,6] = 5

  labels[4,4,4] = 6
  labels[4,6,4] = 7
  labels[6,4,4] = 8
  labels[6,6,4] = 9

  # not connected to anything else
  labels[1,:,:] = 10

  res = cc3d.region_graph(labels, connectivity=26)
  assert res == set([ (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9) ])

  res = cc3d.region_graph(labels, connectivity=18)
  assert res == set()

  res = cc3d.region_graph(labels, connectivity=6)
  assert res == set()

def test_region_graph_18():
  labels = np.zeros( (10, 10, 10), dtype=np.uint32 )

  labels[5,5,5] = 1
  labels[5,6,6] = 2
  labels[5,4,6] = 3
  labels[6,5,6] = 4
  labels[4,5,6] = 5

  labels[5,4,4] = 6
  labels[5,6,4] = 7
  labels[6,5,4] = 8
  labels[4,5,4] = 9

  # not connected to anything else
  labels[1,:,:] = 10

  res = cc3d.region_graph(labels, connectivity=26)
  assert res == set([ 
    (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), 
    (2,4), (2,5), (3,4), (3,5),
    (6,8), (6,9), (7,8), (7,9),
  ])

  res = cc3d.region_graph(labels, connectivity=18)
  assert res == set([ 
    (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), 
    (2,4), (2,5), (3,4), (3,5),
    (6,8), (6,9), (7,8), (7,9),
  ])

  res = cc3d.region_graph(labels, connectivity=6)
  assert res == set()


def test_region_graph_6():
  labels = np.zeros( (10, 10, 10), dtype=np.uint32 )

  labels[5,5,5] = 1

  labels[5,5,6] = 2
  labels[5,5,4] = 3

  labels[4,5,5] = 4
  labels[6,5,5] = 5

  labels[5,4,5] = 6
  labels[5,6,5] = 7

  # not connected to anything else
  labels[1,:,:] = 10

  res = cc3d.region_graph(labels, connectivity=26)
  assert res == set([ 
    (1,2), (1,3), (1,4), (1,5), (1,6), (1,7),
    (2,4), (2,5), (2,6), (2,7),
    (3,4), (3,5), (3,6), (3,7),
    (4,6), (4,7),
    (5,6), (5,7)
  ])

  res = cc3d.region_graph(labels, connectivity=18)
  assert res == set([ 
    (1,2), (1,3), (1,4), (1,5), (1,6), (1,7),
    (2,4), (2,5), (2,6), (2,7),
    (3,4), (3,5), (3,6), (3,7),
    (4,6), (4,7),
    (5,6), (5,7)
  ])

  res = cc3d.region_graph(labels, connectivity=6)
  assert res == set([
    (1,2), (1,3), (1,4), (1,5), (1,6), (1,7)
  ])

def test_voxel_graph_2d():
  labels = np.ones((3,3), dtype=np.uint8)
  graph = cc3d.voxel_connectivity_graph(labels, connectivity=4)
  assert graph.dtype == np.uint8
  assert np.all(graph)

  graph = cc3d.voxel_connectivity_graph(labels, connectivity=8)
  assert graph.dtype == np.uint8
  assert np.all(graph)

  labels[1,1] = 0
  graph = cc3d.voxel_connectivity_graph(labels, connectivity=4)
  gt = np.array([
    [0x0f,       0b00001011, 0x0f      ],
    [0b00001110, 0x00,       0b00001101],
    [0x0f,       0b00000111, 0x0f      ]
  ], dtype=np.uint8)
  assert np.all(gt.T == graph)

  graph = cc3d.voxel_connectivity_graph(labels, connectivity=8)
  gt = np.array([
    [0b11101111, 0b11111011, 0b11011111],
    [0b11111110, 0x00,       0b11111101],
    [0b10111111, 0b11110111, 0b01111111]
  ], dtype=np.uint8)
  assert np.all(gt.T == graph)

def test_voxel_graph_3d():
  labels = np.ones((3,3,3), dtype=np.uint8)
  graph = cc3d.voxel_connectivity_graph(labels, connectivity=26)
  assert graph.dtype == np.uint32
  assert np.all(graph)

  E = {
    (-1,-1,-1): 0b01111111111111111111111111,
    (+1,-1,-1): 0b10111111111111111111111111,
    (-1,+1,-1): 0b11011111111111111111111111,
    (+1,+1,-1): 0b11101111111111111111111111,
    (-1,-1,+1): 0b11110111111111111111111111,
    (+1,-1,+1): 0b11111011111111111111111111,
    (-1,+1,+1): 0b11111101111111111111111111,
    (+1,+1,+1): 0b11111110111111111111111111,
    ( 0,-1,-1): 0b11111111011111111111111111,
    ( 0,+1,-1): 0b11111111101111111111111111,
    (-1, 0,-1): 0b11111111110111111111111111,
    (+1, 0,-1): 0b11111111111011111111111111,
    ( 0,-1,+1): 0b11111111111101111111111111,
    ( 0,+1,+1): 0b11111111111110111111111111,
    (-1, 0,+1): 0b11111111111111011111111111,
    (+1, 0,+1): 0b11111111111111101111111111,
    (-1,-1, 0): 0b11111111111111110111111111,
    (+1,-1, 0): 0b11111111111111111011111111,
    (-1,+1, 0): 0b11111111111111111101111111,
    (+1,+1, 0): 0b11111111111111111110111111,
    ( 0, 0,-1): 0b11111111111111111111011111,
    ( 0, 0,+1): 0b11111111111111111111101111,
    ( 0,-1, 0): 0b11111111111111111111110111,
    ( 0,+1, 0): 0b11111111111111111111111011,
    (-1, 0, 0): 0b11111111111111111111111101,
    (+1, 0, 0): 0b11111111111111111111111110,
  }


  labels[1,1,1] = 0
  graph = cc3d.voxel_connectivity_graph(labels, connectivity=26)
  gt = np.array([
    [
      [ E[(1, 1,1)], E[(0, 1,1)], E[(-1, 1,1)] ],
      [ E[(1, 0,1)], E[(0, 0,1)], E[(-1, 0,1)] ],
      [ E[(1,-1,1)], E[(0,-1,1)], E[(-1,-1,1)] ],
    ],
    [
      [ E[(1, 1,0)], E[(0,1,0)], E[(-1, 1,0)] ],
      [ E[(1, 0,0)], 0x00,       E[(-1, 0,0)] ],
      [ E[(1,-1,0)], E[(0,-1,0)], E[(-1,-1,0)] ]    
    ],
    [
      [ E[(1,1,-1)], E[(0,1,-1)], E[(-1,1,-1)] ],
      [ E[(1,0,-1)], E[(0,0,-1)], E[(-1,0,-1)] ],
      [ E[(1,-1,-1)], E[(0,-1,-1)], E[(-1,-1,-1)] ]
    ]
  ], dtype=np.uint32)
  
  assert np.all(gt.T == graph)


