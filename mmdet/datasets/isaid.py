from .xml_style import XMLDataset
from .registry import DATASETS


@DATASETS.register_module
class ISaidDataset(XMLDataset):

    CLASSES = ('storage_tank', 'Large_Vehicle', 'Small_Vehicle', 'ship', 'Harbor',
               'baseball_diamond', 'Ground_Track_Field', 'Soccer_ball_field', 'Swimming_pool',
               'Roundabout', 'tennis_court', 'basketball_court', 'plane', 'Helicopter', 'Bridge')
