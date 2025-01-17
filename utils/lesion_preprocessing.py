import dpp
import dpp.twodim.segmentation as seg
import dpp.threedim.segmentation.medical as vol


input_identifier = "volume"
ground_truth_identifier = "segmentation"

train_split = [0.25, 0.75]
val_split = [0.85, 0.15]
seed = 42


def training(train_data_dir, save_name=""):
    """
    Creates the pre-processing pipeline used during training. 

    Parameters
    ----------
    train_data_dir: string
        The directory which contains the training and validation data.
        The data is split into a training and validation set using the global seed.
    save_name: string
        The name used to save temporary values needed for pre-processing and specific to a given dataset.
        These can be re-loaded, which greatly speeds up initialization.

    Returns
    -------
    training_pipeline: iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary and
        the first input in the list is the network input, while the second input is the corresponding ground truth.
    """

    global input_identifier
    global ground_truth_identifier

    global split
    global seed

    # Volumes for initialization #

    # init = _get_initialization_generator(train_data_dir)
    # Slices for training #

    with dpp.Pipeline(storage_name=save_name) as pipe:

        # Load slices
        node = dpp.reader.file_paths(train_data_dir, input_identifier=input_identifier, ground_truth_identifier=ground_truth_identifier, random=True, iterations=0, split=train_split, split_selector=0, seed=seed)
        # node = seg.medical.decreasing_load_slice_filtered(node, label_of_interest=2, label_required=1, min_frequency=0.8, max_tries=20, slice_type='axial', depth=3, single_label_slice=False)
        node = seg.medical.load_slice_filtered(node, label_of_interest=2, label_required=1, min_frequency=0.88, max_tries=300, slice_type='axial', depth=5, single_label_slice=False)
        # node = seg.medical.modified_load_all_slices(node, label_required=1, label_required_occ_rate=0.00, label_void_occ_rate=0.0, slice_type='axial', depth=5, single_label_slice=False)

        # Random rotation then crop to fit
        node = seg.random_rotation(node, probability=0.1, upper_bound=180)
        # node = seg.crop_to_label(node, 1, max_reduction=10, square_crop=False, default_label=0)

        # Random transformations
        # node = seg.random_resize(node, [312, 312], probability=0.1, lower_bound=0.8, upper_bound=1.1, default_pixel=0., default_label=0)
        # node = seg.resize(node, [312, 312])
        node = seg.random_translation(node, probability=0.1, border_usage=0.5, default_border=0.25, label_of_interest=1, default_pixel=0., default_label=0)
        # node = seg.elastic_transformation(node, probability=0.25, coef_alpha=None, coef_sigma=None, coef_alpha_affine=None)

        # Adjust colours and labels
        # node = seg.decreasing_mask_img_background(node, 0, pixel_value=0.)
        node = seg.mask_img_background(node, 0, pixel_value=0.)
        node = seg.reduce_to_single_label_slice(node)
        # node = seg.decreasing_clip_img_values(node, 0, 200.)
        node = seg.clip_img_values(node, 0., 200)
        node = seg.fuse_labels_other_than(node, 2)

        # Prepare as input
        node = seg.transpose(node)

        #node = seg.robust_img_scaling(node, ignore_values=[0., 200.], initialize=True)

    return pipe


def validation(validation_data_dir, save_name=""):
    """
    Creates the pre-processing pipeline used during validation. 

    Parameters
    ----------
    validation_data_dir: string
        The directory which contains the training and validation data.
        The data is split into a training and validation set using the global seed.
    save_name: string
        The name used to save temporary values needed for pre-processing and specific to a given dataset.
        These can be re-loaded, which greatly speeds up initialization.

    Returns
    -------
    validation_pipeline: iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary and
        the first input in the list is the network input, while the second input is the corresponding ground truth.
    """

    global input_identifier
    global ground_truth_identifier

    global split
    global seed

    # Volumes for initialization #
    # init = _get_initialization_generator(validation_data_dir)

    with dpp.Pipeline(storage_name=save_name) as pipe:

        node = dpp.reader.file_paths(validation_data_dir, input_identifier=input_identifier, ground_truth_identifier=ground_truth_identifier, random=False, iterations=0, split=val_split, split_selector=1, seed=seed)
        node = seg.medical.load_all_slices_filtered(node, label_required=1, slice_type='axial', depth=5, single_label_slice=False)
        # node = seg.medical.decreasing_load_all_slices(node, slice_type='axial', depth=3, single_label_slice=False)
        node = _test_val_tail(node)

    return pipe


def test(test_data_dir, save_name=""):
    """
    Creates the pre-processing pipeline used during testing. 

    Parameters
    ----------
    test_data_dir: string
        The directory which contains the test data.
    save_name: string
        The name used to save temporary values needed for pre-processing and specific to a given dataset.
        These can be re-loaded, which greatly speeds up initialization.

    Returns
    -------
    test_pipeline: iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary and
        the first input in the list is the network input, while the second input is the corresponding ground truth.
    """

    global input_identifier
    global ground_truth_identifier

    # Volumes for initialization #
    init = _get_initialization_generator(test_data_dir)

    with dpp.Pipeline(storage_name=save_name, initialization_generator=init) as pipe:

        node = dpp.reader.file_paths(test_data_dir, input_identifier="test-volume", ground_truth_identifier="", random=False, iterations=1)
        node = seg.medical.load_all_slices(node, slice_type='axial', depth=5, single_label_slice=False)
        node = _test_val_tail(node)

    return pipe


def generate_predictions(test_data_dir, save_name=""):
    """
    Creates the pre-processing pipeline used when generating predictions. 

    Parameters
    ----------
    test_data_dir: string
        The directory which contains the test data.
    save_name: string
        The name used to save temporary values needed for pre-processing and specific to a given dataset.
        These can be re-loaded, which greatly speeds up initialization.

    Returns
    -------
    test_pipeline: iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary and
        the first input in the list is the network input, while the second input is the corresponding ground truth.
    """

    # Volumes for initialization #
    # init = _get_initialization_generator(test_data_dir)

    with dpp.Pipeline(storage_name=save_name) as pipe:

        node = dpp.reader.file_paths(test_data_dir, input_identifier="test-volume-", ground_truth_identifier="-ori", random=False, iterations=1)
        node = seg.medical.load_all_slices(node, slice_type='axial', depth=5, single_label_slice=False)
        
        # Adjust labels and colours
        node = seg.crop_to_label(node, 1, max_reduction=10, square_crop=True, default_label=0)
        node = seg.resize(node, [312, 312])

        node = seg.mask_img_background(node, 0, pixel_value=0.)
        node = seg.reduce_to_single_label_slice(node)
        node = seg.clip_img_values(node, 0., 200.)

        # Prepare as network input
        # node = seg.resize(node, [256, 256])
        node = seg.transpose(node)
        # node = seg.pad(node, 92, mode='constant', image_only=True, default_pixel=0.)

        # node = seg.robust_img_scaling(node, ignore_values=[0., 200.], initialize=True)

    return pipe


def _test_val_tail(node):

    # Adjust labels and colours
    # node = seg.crop_to_label(node, 1, max_reduction=10, square_crop=True, default_label=0)
    # node = seg.resize(node, [312, 312])

    # node = seg.decreasing_mask_img_background(node, 0, pixel_value=0.)
    node = seg.mask_img_background(node, 0, pixel_value=0.)
    node = seg.reduce_to_single_label_slice(node)
    node = seg.clip_img_values(node, 0, 200.)
    # node = seg.decreasing_clip_img_values(node, 0, 200.)
    node = seg.fuse_labels_other_than(node, 2)

    # Prepare as network input
    # node = seg.resize(node, [256, 256])
    node = seg.transpose(node)

    # node = seg.robust_img_scaling(node, ignore_values=[0., 200.], initialize=True)

    return node


def _get_initialization_generator(data_dir):
    """
    Creates the pre-processing pipeline used during initialization of the other pre-processing pipelines. 

    Parameters
    ----------
    data_dir: string
        The directory which contains the respective data.

    Returns
    -------
    data_pipeline: iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary and
        the first input in the list is the network input, while the second input is the corresponding ground truth.
    """

    global input_identifier
    global ground_truth_identifier

    with dpp.Pipeline() as init:
        node = dpp.reader.file_paths(data_dir, input_identifier=input_identifier, ground_truth_identifier=ground_truth_identifier, random=False, iterations=1)
        node = vol.load_volume(node)
        node = seg.mask_img_background(node, 0, pixel_value=0.)
        node = seg.clip_img_values(node, 0., 200.)

    return init
