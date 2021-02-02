import os


def eval_cityscapes(res_dir):
    os.environ['CITYSCAPES_DATASET'] = '/store/datasets/cityscapes'
    os.environ['CITYSCAPES_RESULTS'] = res_dir
    from cityscapesscripts.evaluation import evalInstanceLevelSemanticLabeling
    # os.system('pwd')
    # os.system('python cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py')
    print('AP: ', evalInstanceLevelSemanticLabeling.getAP())

eval_cityscapes('/usagers2/huper/dev/CenterPoly/exp/cityscapes/polydet/hg_32pts/results')

    