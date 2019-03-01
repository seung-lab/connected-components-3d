import pytest

import cc3d
import numpy as np

TEST_TYPES = [
  np.int8, np.int16, np.int32, np.int64,
  np.uint8, np.uint16, np.uint32, np.uint64,
]

def gt_c2f(gt):
  f_gt = np.copy(gt)
  mx = np.max(gt) + 1
  f_gt[ f_gt == 2 ] = mx
  f_gt[ f_gt == 3 ] = 2
  f_gt[ f_gt == mx ] = 3
  return f_gt

def test_2d_square():
  def test(order, ground_truth):
    for dtype in TEST_TYPES:
      input_labels = np.zeros( (16,16), dtype=dtype, order=order )
      input_labels[:8,:8] = 8
      input_labels[8:,:8] = 9
      input_labels[:8,8:] = 10
      input_labels[8:,8:] = 11

      output_labels = cc3d.connected_components(input_labels).astype(dtype)
      
      print(order)
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

def test_2d_rectangle():
  def test(order, ground_truth):
    for dtype in TEST_TYPES:
      input_labels = np.zeros( (16,13,1), dtype=dtype, order=order )
      input_labels[:8,:8,:] = 8
      input_labels[8:,:8,:] = 9
      input_labels[:8,8:,:] = 10
      input_labels[8:,8:,:] = 11

      output_labels = cc3d.connected_components(input_labels).astype(dtype)
      print(output_labels.shape)
      output_labels = output_labels[:,:,0]

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

def test_2d_cross():
  def test(order, ground_truth):
    for dtype in TEST_TYPES:
      input_labels = np.zeros( (17,17), dtype=dtype, order=order)
      input_labels[:] = 1
      input_labels[:,8] = 0
      input_labels[8,:] = 0

      output_labels = cc3d.connected_components(input_labels).astype(dtype)
      print(output_labels)

      assert np.all(output_labels == ground_truth)

      input_labels[9:,9:] = 2
      output_labels = cc3d.connected_components(input_labels).astype(dtype)
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


def test_2d_cross_with_intruder():
  def test(order, ground_truth):
    input_labels = np.zeros( (5,5), dtype=np.uint8, order=order)
    input_labels[:] = 1
    input_labels[:,2] = 0
    input_labels[2,:] = 0
    input_labels[3:,3:] = 2
    input_labels[3,3] = 1

    output_labels = cc3d.connected_components(input_labels).astype(np.uint8)
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

def test_3d_all_different():
  for order in ("C", "F"):
    input_labels = np.arange(0, 100 * 99 * 98).astype(np.uint32)
    input_labels = input_labels.reshape((100,99,98), order=order)

    output_labels = cc3d.connected_components(input_labels)

    assert np.unique(output_labels).shape[0] == 100*99*98
    assert output_labels.shape == (100, 99, 98)

def test_3d_cross():
  def test(order, ground_truth):
    print(order)
    for dtype in TEST_TYPES:
      print(dtype)
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

def test_3d_cross_asymmetrical():
  def test(order, ground_truth):
    print(order)
    for dtype in TEST_TYPES:
      print(dtype)
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

# def test_sixty_four_bit():
#   input_labels = np.ones((1626,1626,1626), dtype=np.uint8)
#   cc3d.connected_components(input_labels, max_labels=0)  
