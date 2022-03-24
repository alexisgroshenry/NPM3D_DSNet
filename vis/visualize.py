#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
from types import new_class
import yaml
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscanvis import LaserScanVis

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./visualize.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=False,
      default='../data',
      help='Dataset to visualize. No Default',
  )
  parser.add_argument(
      '--config', '-c',
      type=str,
      required=False,
      default="../semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--sequence', '-s',
      type=str,
      default="00",
      required=False,
      help='Sequence to visualize. Defaults to %(default)s',
  )
  parser.add_argument(
      '--predictions', '-p',
      type=str,
      default=None,
      required=False,
      help='Alternate location for labels, to use predictions folder. '
      'Must point to directory containing the predictions in the proper format '
      ' (see readme)'
      'Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_semantics', '-i',
      dest='ignore_semantics',
      default=False,
      action='store_true',
      help='Ignore semantics. Visualizes uncolored pointclouds.'
      'Defaults to %(default)s',
  )
  parser.add_argument(
      '--do_instances', '-di',
      dest='do_instances',
      default=False,
      action='store_true',
      help='Visualize instances too. Defaults to %(default)s',
  )
  parser.add_argument(
      '--gt_and_preds', '-gp',
      dest='gt_and_preds',
      default=False,
      action='store_true',
      help='Visualize ground truth and predictions alltogether.If --do instance is false, the semantic GT and predictions will be displayed, otherwise instance GT and results.',
  )
  parser.add_argument(
      '--offset',
      type=int,
      default=0,
      required=False,
      help='Sequence to start. Defaults to %(default)s',
  )
  parser.add_argument(
      '--size',
      type=int,
      default=1,
      required=False,
      help='Size of points. Defaults to %(default)s',
  )
  parser.add_argument(
      '--plot_legend', '-pl',
      dest='plot_legend',
      default=False,
      action='store_true',
      help='Plot color legend for the semantic classes',
  )
  parser.add_argument(
      '--bgcolor',
      type=str,
      default='black',
      required=False,
      help='Background color. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_safety',
      dest='ignore_safety',
      default=False,
      action='store_true',
      help='Normally you want the number of labels and ptcls to be the same,'
      ', but if you are not done inferring this is not the case, so this disables'
      ' that safety.'
      'Defaults to %(default)s',
  )
  FLAGS, unparsed = parser.parse_known_args()

  if FLAGS.gt_and_preds and FLAGS.predictions==None:
    raise RuntimeError("No prediction file received for argument --prediction")

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Dataset", FLAGS.dataset)
  print("Config", FLAGS.config)
  print("Sequence", FLAGS.sequence)
  print("Predictions", FLAGS.predictions)
  print("ignore_semantics", FLAGS.ignore_semantics)
  print("do_instances", FLAGS.do_instances)
  print("gt_and_preds", FLAGS.gt_and_preds)
  print("ignore_safety", FLAGS.ignore_safety)
  print("offset", FLAGS.offset)
  print("size", FLAGS.size)
  print("*" * 80)

  # open config file
  try:
    print("Opening config file %s" % FLAGS.config)
    CFG = yaml.safe_load(open(FLAGS.config, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # fix sequence name
  FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

  # does sequence folder exist?
  scan_paths = os.path.join(FLAGS.dataset, "sequences",
                            FLAGS.sequence, "velodyne")
  if os.path.isdir(scan_paths):
    print("Sequence folder exists! Using sequence from %s" % scan_paths)
  else:
    print("Sequence folder doesn't exist! Exiting...")
    quit()

  # populate the pointclouds
  scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(scan_paths)) for f in fn]
  scan_names.sort()

  # does sequence folder exist?
  if not FLAGS.ignore_semantics:
    gt_paths = os.path.join(FLAGS.dataset, "sequences",
                                 FLAGS.sequence, "labels")
    if os.path.isdir(gt_paths):
      print("GT labels folder exists! Using GT labels from %s" % gt_paths)
    else:
      print("GT labels folder doesn't exist! Exiting...")
      quit()
    # populate the pointclouds
    gt_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(gt_paths)) for f in fn]
    gt_names.sort()

    if FLAGS.predictions is not None:
      pred_paths = os.path.join(FLAGS.predictions, "sequences",
                                 FLAGS.sequence, "predictions")
      if os.path.isdir(gt_paths):
        print("Predicted Labels folder exists! Using predicted labels from %s" % pred_paths)
      else:
        print("Predicted labels folder doesn't exist! Exiting...")
        quit()
      pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(pred_paths)) for f in fn]
      pred_names.sort()
    else:
      pred_paths = None
      pred_names=None
    
    # check that there are same amount of labels and scans
    if not FLAGS.ignore_safety:
      assert(len(gt_names) == len(scan_names))
      assert(pred_paths==None or len(pred_names) == len(scan_names))
  
  # create a scan
  if FLAGS.ignore_semantics:
    scan = LaserScan(project=True)  # project all opened scans to spheric proj
  else:
    color_dict = CFG["color_map"]
    colors = [tuple([c2/255. for c2 in c][::-1]) for c in color_dict.values()][2:30]
    nclasses = len(color_dict)
    if FLAGS.gt_and_preds and FLAGS.do_instances:
      pred_type = 'inst'
    elif FLAGS.gt_and_preds:
      pred_type = 'sem'
    else:
      pred_type = None
    scan = SemLaserScan(nclasses, color_dict, project=True, preds=pred_type)

  if FLAGS.plot_legend:
    import numpy as np
    from matplotlib import pyplot as plt
    fig = plt.figure("Line plot")
    legendFig = plt.figure("Legend plot")
    ax = fig.add_subplot(111)
    points=[]
    for i in range(28):
      points.append(ax.scatter(i,0,c=colors[i],s=5))
    legendFig.legend(points, list(CFG['labels'].values())[2:30], loc='center')
    legendFig.show()

  # create a visualizer
  semantics = not FLAGS.ignore_semantics
  instances = FLAGS.do_instances
  if not semantics:
    gt_names = None
    pred_names=None


  vis = LaserScanVis(scan=scan,
                     scan_names=scan_names,
                     gt_names=gt_names,
                     pred_names=pred_names,
                     offset=FLAGS.offset,
                     size=FLAGS.size,
                     semantics=semantics,
                     instances=instances and semantics,
                     gt_and_preds=FLAGS.gt_and_preds
                     )

  # print instructions
  print("To navigate:")
  print("\tb: back       (previous scan)")
  print("\tn: next       (next scan)")
  print("\tp: plus       (increase point size)")
  print("\tm: minus      (decrease point size)")
  print("\td: difference (difference between GT and preds)")
  print("\tq: quit       (exit program)")

  # run the visualizer
  vis.run()
