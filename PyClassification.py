from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import cStringIO as StringIO
import urllib2
import math
import exifutil
import tensorflow as tf
# from inception import inception_model
from PIL import Image
from fileinput import filename

import argparse
import sys

REPO_DIRNAME = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + '/data')
NUM_CLASSES = 5
NUM_TOP_CLASSES = 5
# Obtain the flask app object
app = flask.Flask(__name__)
UPLOAD_FOLDER = '/tmp/demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'jpg', 'jpe', 'jpeg'])


@app.route('/', methods=['GET', 'POST'])
def classify_index():
  string_buffer = None

  if flask.request.method == 'GET':
    url = flask.request.args.get('url')
    if url:
      logging.info('Image: %s', url)
      string_buffer = urllib2.urlopen(url).read()

    file = flask.request.args.get('file')
    if file:
      logging.info('Image: %s', file)
      string_buffer = open(file, 'rb').read()

    if not string_buffer:
      return flask.render_template('index.html', has_result=False)

  elif flask.request.method == 'POST':
    string_buffer = flask.request.stream.read()

  if not string_buffer:
    resp = flask.make_response()
    resp.status_code = 400
    return resp
  names, probs, time_cost, accuracy = app.clf.classify_image(string_buffer)
  return flask.make_response(u','.join(names), 200,
                             {'ClassificationAccuracy': accuracy})


@app.route('/classify_url', methods=['GET'])
def classify_url():
  imageurl = flask.request.args.get('imageurl', '')
  try:
    bytes = urllib2.urlopen(imageurl).read()
    string_buffer = StringIO.StringIO(bytes)
    image = exifutil.open_oriented_im(string_buffer)

  except Exception as err:
    # For any exception we encounter in reading the image, we will just
    # not continue.
    logging.info('URL Image open error: %s', err)
    return flask.render_template(
        'index.html',
        has_result=True,
        result=(False, 'Cannot open image from URL.'))

  app.logger.info('Image: %s', imageurl)
  names, probs, time_cost, accuracy = app.clf.classify_image(bytes)
  return flask.render_template(
      'index.html',
      has_result=True,
      result=[True, zip(names, probs),
              '%.3f' % time_cost],
      imagesrc=embed_image_html(image))


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
  try:
    # We will save the file to disk for possible data collection.
    imagefile = flask.request.files['imagefile']
    filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
        werkzeug.secure_filename(imagefile.filename)
    filename = os.path.join(UPLOAD_FOLDER, filename_)
    imagefile.save(filename)
    path, extension = os.path.splitext(filename)
    if extension == '.png':
      im = Image.open(filename)
      filename = '%s.jpg' % path
      im.save(filename)

    logging.info('Saving to %s.', filename)
    image = exifutil.open_oriented_im(filename)

  except Exception as err:
    logging.info('Uploaded image open error: %s', err)
    return flask.render_template(
        'index.html',
        has_result=True,
        result=(False, 'Cannot open uploaded image.'))

  names, probs, time_cost, accuracy = app.clf.classify_image(
      open(os.path.join(filename), 'rb').read())
  return flask.render_template(
      'index.html',
      has_result=True,
      result=[True, zip(names, probs),
              '%.3f' % time_cost],
      imagesrc=embed_image_html(image))


def embed_image_html(image):
  """Creates an image embedded in HTML base64 format."""
  image_pil = Image.fromarray((255 * image).astype('uint8'))
  image_pil = image_pil.resize((256, 256))
  string_buf = StringIO.StringIO()
  image_pil.save(string_buf, format='png')
  data = string_buf.getvalue().encode('base64').replace('\n', '')
  return 'data:image/png;base64,' + data


def allowed_file(filename):
  return ('.' in filename and
          filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS)


class ImagenetClassifier(object):
  default_args = {
      'model_def_file': ('/tmp/frozen_inception_v3_63425.pb'),
      'class_labels_file': (os.path.join(REPO_DIRNAME, 'labels.txt')),
  }
  for key, val in default_args.iteritems():
    if not os.path.exists(val):
      raise Exception(
          'File for {} is missing. Should be at: {}'.format(key, val))

  def _load_graph(self, model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)

    return graph

  def _read_tensor_from_image_file(self, file_buffer, input_height=299,
                                   input_width=299,
                                   input_mean=0, input_std=255):
    input_name = "file_buffer"
    output_name = "normalized"

    image_reader = tf.image.decode_jpeg(file_buffer, channels=3,
                                        name='jpeg_reader')
    # if file_name.endswith(".png"):
    #   image_reader = tf.image.decode_png(file_buffer, channels=3,
    #                                      name='png_reader')
    # else:
    # elif file_name.endswith(".gif"):
    #   image_reader = tf.squeeze(tf.image.decode_gif(file_buffer,
    #                                                 name='gif_reader'))
    # elif file_name.endswith(".bmp"):
    #   image_reader = tf.image.decode_bmp(file_buffer, name='bmp_reader')
    #                                       name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    central_crop = tf.image.central_crop(float_caster, central_fraction=0.875)
    dims_expander = tf.expand_dims(central_crop, 0);
    resized = tf.image.resize_bilinear(dims_expander,
                                       [input_height, input_width],
                                       align_corners=False)
    normalized = tf.multiply(tf.subtract(resized, 0.5), 2.0)
    sess = tf.Session()
    result = sess.run(normalized)
    return result

  def _load_labels(self, label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
      label.append(l.rstrip())
    return label

  def __init__(self, model_def_file, class_labels_file):
    logging.info('Loading net and associated files...')

    self._graph = self._load_graph(model_def_file)
    self._label_names = self._load_labels(class_labels_file)
    input_layer = 'input'
    output_layer = 'InceptionV3/Predictions/Reshape_1'
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    self._input_operation = self._graph.get_operation_by_name(input_name);
    self._output_operation = self._graph.get_operation_by_name(output_name);

    # with tf.Graph().as_default(), tf.device('cpu:0'):
    #   self.sess = tf.Session()
    #   self.image_tensor = tf.placeholder(tf.float32, (299, 299, 3))

    #   input_name = "import/" + input_layer
    #   output_name = "import/" + output_layer
    #   input_operation = self._graph.get_operation_by_name(input_name);
    #   output_operation = self._graph.get_operation_by_name(output_name);

    #   image_tensor = _read_tensor_from_image_file(self.image_buffer)
    #   # Run inference.
    #   logits, predictions = inception_model.inference(images, NUM_CLASSES + 1)
    #   with tf.Session(graph=graph) as sess:
    #     with tf.device('/cpu:0'):
    #       results = sess.run(output_operation.outputs[0],
    #                          {input_operation.outputs[0]: t})

    #   results = np.squeeze(results)
    #   top_k = results.argsort()[-5:][::-1]

    #   self.label_names = ['none']
    #   for i in top_k:
    #     self.label_names.append(self._label[i])

  def eval_image(self, image, height, width, scope=None):
    """Prepare one image for evaluation.

        Args:
          image: 3-D float Tensor
          height: integer
          width: integer
          scope: Optional scope for op_scope.
        Returns:
          3-D float Tensor of prepared image.
        """
    #image = tf.reshape(image, [height,width,3])
    # return images
    with tf.op_scope([image, height, width], scope, 'eval_image'):
      # Crop the central region of the image with an area containing 87.5% of
      # the original image.
      image = tf.image.central_crop(image, central_fraction=0.875)

      # Resize the image to the original height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(
          image, [height, width], align_corners=False)
      image = tf.squeeze(image, [0])
      return image

  def classify_image(self, image):
    try:
      start_time = time.time()
      image_tensor = self._read_tensor_from_image_file(image)
      # Run inference.
      config = tf.ConfigProto(
        device_count= {'GPU': 0}
      )
      with tf.Session(graph=self._graph, config=config) as sess:
        results = sess.run(self._output_operation.outputs[0],
                           {self._input_operation.outputs[0]: image_tensor})
      app.logger.info('classification results: %s' % results)
      results = np.squeeze(results)
      top_k = results.argsort()[-5:][::-1]
      end_time = time.time()
      labels = []
      probs = []
      for i in top_k:
        labels.append(i)
        probs.append(results[i])

      app.logger.info('classify_image cost %.2f secs', end_time - start_time)
      return [
          self._label_names[labels[0]], self._label_names[labels[1]],
          self._label_names[labels[2]]
      ], probs[:3], end_time - start_time, sum(probs)
    except Exception as err:
      logging.info('Classification error: %s', err)
      return None


def setup_app(app):
  app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
  app.logger.info('testing sample picture...')
  bytes = open(os.path.join(REPO_DIRNAME, 'sample.jpg'), 'rb').read()
  ret, _, _, _ = app.clf.classify_image(bytes)
  app.logger.info('sample testing complete %s %s %s', ret[0], ret[1], ret[2])


def start_from_terminal(app):
  """
    Parse command line options and start the server.
    """
  parser = optparse.OptionParser()
  parser.add_option(
      '-p',
      '--port',
      help='which port to serve content on',
      type='int',
      default=5005)
  opts, args = parser.parse_args()
  # Initialize classifier + warm start by forward for allocation
  setup_app(app)
  app.run(debug=True, processes=1, host='0.0.0.0', port=opts.port)


logging.getLogger().setLevel(logging.INFO)
if not os.path.exists(UPLOAD_FOLDER):
  os.makedirs(UPLOAD_FOLDER)

if __name__ == '__main__':
  start_from_terminal(app)
else:
  gunicorn_error_logger = logging.getLogger('gunicorn.error')
  app.logger.handlers = gunicorn_error_logger.handlers
  app.logger.setLevel(logging.INFO)
  setup_app(app)
