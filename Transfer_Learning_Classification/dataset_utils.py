import argparse
import os
import sys
import glob
import pickle
import tensorflow as tf
import numpy as np
import sys
import six
import imghdr
from PIL import Image

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if six.PY3 and isinstance(value, six.text_type):           
        value = six.binary_type(value, encoding='utf-8') 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def deserialize_image_record(record):
    feature_map = {
            'label': tf.FixedLenFeature([ ], tf.int64,-1),
            'text_label': tf.FixedLenFeature([ ], tf.string, ''),
            'image': tf.FixedLenFeature([ ], tf.string, ''),
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image']
        label   = tf.cast(obj['label'], tf.int32)
        text_label =  obj['text_label']
        return imgdata, label, text_label


def convert_to_tfrecord(dataset_name, data_directory, class_map, segments=1, directories_as_labels=True, files='**/*.jpg'):
    """Convert the dataset into TFRecords on disk
    
    Args:
        dataset_name:   The name/folder of the dataset
        data_directory: The directory where records will be stored
        class_map:      Dictionary mapping dictory label name to integer label
        segments:       The number of files on disk to separate records into
        directories_as_labels: Whether the directory name should be used as it's label (used for test directory)
        files:          Which files to find in the data directory
    """
    
    # Create a dataset of file path and class tuples for each file
    
    dataset_directory = os.path.join(data_directory, dataset_name)
    filenames = glob.glob(os.path.join(dataset_directory, files))
    classes = (os.path.basename(os.path.dirname(name)) for name in filenames) if directories_as_labels else [None] * len(filenames)
    dataset = list(zip(filenames, classes))
    
    # If sharding the dataset, find how many records per file
    num_examples = len(filenames)
    samples_per_segment = num_examples // segments
    
    print("Have {} per record file".format(samples_per_segment))
    
    for segment_index in range(segments):
        start_index = segment_index * samples_per_segment
        end_index = (segment_index + 1) * samples_per_segment

        sub_dataset = dataset[start_index:end_index]
        record_filename = os.path.join(data_directory, '{}-{}.tfrecords'.format(dataset_name, segment_index))

        with tf.python_io.TFRecordWriter(record_filename) as writer:
            print("Writing {}".format(record_filename))

            for index, sample in enumerate(sub_dataset):
                sys.stdout.write("\rProcessing sample {} of {}".format(start_index+index+1, num_examples))
                sys.stdout.flush()
                
                file_path, label = sample
                image = None
                if imghdr.what(file_path) != 'jpeg':
                    print(imghdr.what(file_path))
                    raise ValueError('{} is not a jpeg!'.format(file_path))
                    
                with tf.gfile.FastGFile(file_path, 'rb') as f:
                    image_data = f.read()

                encoded_label = None
                if label:
                    encoded_label = str.encode(label)

                features = {
                    'label': _int64_feature(class_map[label]),
                    'text_label': _bytes_feature(encoded_label),
                    'image': _bytes_feature(image_data)
                }

                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())   


def process_directory(data_directory):
    """Process the directory to convert images to TFRecords"""

    data_dir = os.path.expanduser(data_directory)
    train_data_dir = os.path.join(data_dir, 'train')
    print(train_data_dir)

    class_names = os.listdir(train_data_dir) # Get names of classes
    class_name2id = { label: index for index, label in enumerate(class_names) } # Map class names to integer labels

    print('Dataset label map: {}'.format(class_name2id))

    # Persist this mapping so it can be loaded when training for decoding
    with open(os.path.join(data_directory, 'class_name2id.p'), 'wb') as p:
        pickle.dump(class_name2id, p, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('Converting train dataset from: {}'.format(os.path.join(data_dir, 'train')))
    convert_to_tfrecord('train', data_dir, class_name2id, segments=1)
    print('Converting validation dataset from: {}'.format(os.path.join(data_dir, 'validation')))
    convert_to_tfrecord('validation', data_dir, class_name2id)
    print('Converting test dataset from: {}'.format(os.path.join(data_dir, 'test')))     
    convert_to_tfrecord('test', data_dir, class_name2id)

    
