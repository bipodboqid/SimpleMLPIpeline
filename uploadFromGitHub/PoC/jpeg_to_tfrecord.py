
import tensorflow as tf
import os

def _list_all_files_exclude_parent(directory):
    file_list = []
    # 親ディレクトリの深さを計算
    parent_depth = directory.rstrip(os.sep).count(os.sep)
    
    for dirpath, dirnames, filenames in os.walk(directory):
        # 親ディレクトリ直下のファイル、および孫ディレクトリのファイルはスキップ
        current_depth = dirpath.count(os.sep)
        if current_depth != parent_depth + 1:
            continue
        
        # ファイルをリストに追加
        for filename in filenames:
            file_list.append(os.path.join(dirpath, filename))
    
    return file_list

def _generate_label_from_path(image_path):
    label = str(image_path).split(sep=os.sep)[-2]
    return label

def _byte_to_bytes_feature(value):
    """Returns a bytes_list from a byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _string_to_bytes_feature(value):
    """Returns a bytes_list from a string."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def convert_images_to_tfrecord(image_root, tfrecord_filename):
    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        for image_path in _list_all_files_exclude_parent(image_root):
            try:
                raw_file = tf.io.read_file(image_path)
            except FileNotFoundError:
                print('Could not find file {}'.format(image_path))
                continue
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _byte_to_bytes_feature(raw_file.numpy()),
                'label': _string_to_bytes_feature(_generate_label_from_path(image_path))
            }))
            writer.write(example.SerializeToString())
