# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 23 2014
"""
A set of factory function to help create usual cases of coordinator
"""


from FilterGenerator import FilterGenerator, Finite3SameFilter
from FilterHolder import customFinite3sameFilter
from Convolver import RGBConvolver
from SubWindowExtractor import (MultiSWExtractor, SubWindowExtractor,
                                FixTargetSWExtractor)
from NumberGenerator import OddUniformGenerator, NumberGenerator
from FeatureExtractor import ImageLinearizationExtractor
from Pooler import (IdentityPooler, MultiPooler, ConvMinPooler,
                    ConvAvgPooler, ConvMaxPooler)
from Aggregator import AverageAggregator, MaximumAggregator, MinimumAggregator
from ConvolutionalExtractor import ConvolutionalExtractor
from Coordinator import (RandConvCoordinator, PixitCoordinator,
                         CompressRandConvCoordinator)
from Logger import StandardLogger, ProgressLogger
from Compressor import SamplerFactory, PCAFactory


__all__ = ["Const", "coordinatorRandConvFactory", "coordinatorPixitFactory",
           "coordinatorCompressRandConvFactory"]


class Const:
    POOLING_NONE = 0
    POOLING_AGGREG_MIN = 1
    POOLING_AGGREG_AVG = 2
    POOLING_AGGREG_MAX = 3
    POOLING_CONV_MIN = 4
    POOLING_CONV_AVG = 5
    POOLING_CONV_MAX = 6


def coordinatorPixitFactory(
        nbSubwindows=10,
        subwindowMinSizeRatio=0.5, subwindowMaxSizeRatio=1.,
        subwindowTargetWidth=16, subwindowTargetHeight=16,
        fixedSize=False,
        subwindowInterpolation=SubWindowExtractor.INTERPOLATION_BILINEAR,
        nbJobs=-1, verbosity=10, tempFolder=None,
        random=True):
    """
    Factory method to create :class:`PixitCoordinator`

    Parameters
    ----------
    nbSubwindows : int >= 0 (default : 10)
        The number of subwindow to extract
    subwindowMinSizeRatio : float > 0 (default : 0.5)
        The minimum size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowMaxSizeRatio : float : subwindowMinSizeRatio
    <= subwindowMaxSizeRatio <= 1 (default : 1.)
        The maximim size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowTargetWidth : int > 0 (default : 16)
        The width of the subwindows after reinterpolation
    subwindowTargetHeight : int > 0 (default : 16)
        The height of the subwindows after reinterpolation
    fixedSize : boolean (default : False)
        Whether to use fixe size subwindow. If False, subwindows are drawn
        randomly. If True, the target size is use as the subwindow size and
        only the position is drawn randomly
    subwindowInterpolation : int (default :
    SubWindowExtractor.INTERPOLATION_BILINEAR)
        The subwindow reinterpolation algorithm. For more information, see
        :class:`SubWindowExtractor`

    nbJobs : int >0 or -1 (default : -1)
        The number of process to spawn for parallelizing the computation.
        If -1, the maximum number is selected. See also :mod:`Joblib`.
    verbosity : int >= 0 (default : 10)
        The verbosity level
    tempFolder : string (directory path) (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :class:`ParallelCoordinator`)
    random : bool (default : True)
        Whether to use randomness or use a predefined seed

    Return
    ------
        coordinator : :class:`Coordinator`
            The PixitCoordinator (possibly decorated) corresponding to the set
            of parameters
    Notes
    -----
    - Subwindow random generator
        The subwindow random generator is a :class:`NumberGenerator` base
        instance (generate real nubers uniformely).
    - Feature extractor
        Base instance of :class:`ImageLinearizationExtractor`
    """

    swngSeed = 0
    #Randomness
    if random:
        swngSeed = None
    #SubWindowExtractor
    swNumGenerator = NumberGenerator(seed=swngSeed)
    if fixedSize:
        swExtractor = FixTargetSWExtractor(subwindowTargetWidth,
                                           subwindowTargetHeight,
                                           subwindowInterpolation,
                                           swNumGenerator)
    else:
        swExtractor = SubWindowExtractor(subwindowMinSizeRatio,
                                         subwindowMaxSizeRatio,
                                         subwindowTargetWidth,
                                         subwindowTargetHeight,
                                         subwindowInterpolation,
                                         swNumGenerator)

    multiSWExtractor = MultiSWExtractor(swExtractor, nbSubwindows, True)

    #FEATURE EXTRACTOR
    featureExtractor = ImageLinearizationExtractor()

    #LOGGER
    autoFlush = verbosity >= 45
    logger = ProgressLogger(StandardLogger(autoFlush=autoFlush,
                                           verbosity=verbosity))

    #COORDINATOR
    coordinator = PixitCoordinator(multiSWExtractor, featureExtractor, logger,
                                   verbosity)

    if nbJobs != 1:
        coordinator.parallelize(nbJobs, tempFolder)
    return coordinator


def coordinatorRandConvFactory(
        nbFilters=5,
        filterMinVal=-1, filterMaxVal=1,
        filterMinSize=1, filterMaxSize=17,
        filterNormalisation=FilterGenerator.NORMALISATION_MEANVAR,
        poolings=[(3, 3, Const.POOLING_AGGREG_AVG)],
        nbSubwindows=10,
        subwindowMinSizeRatio=0.5, subwindowMaxSizeRatio=1.,
        subwindowTargetWidth=16, subwindowTargetHeight=16,
        subwindowInterpolation=SubWindowExtractor.INTERPOLATION_BILINEAR,
        includeOriginalImage=False,
        nbJobs=-1, verbosity=10, tempFolder=None,
        random=True):
    """
    Factory method to create :class:`RandConvCoordinator` tuned for RGB images

    Parameters
    ----------
    nbFilters : int >= 0 (default : 5)
        The number of filter
    filterMinVal : float (default : -1)
        The minimum value of a filter component
    filterMaxVal : float : filterMinVal <= filterMaxVal (default : 1)
        The maximum value of a filter component
    filterMinSize : int >= 0 : odd number (default : 1)
        The minimum size of a filter
    filterMaxSize : int >= 0 : odd number s.t.  filterMinSize <= filterMaxSize
    (default : 1)
        The maximum size of a filter
    filterNormalisation : int (default : FilterGenerator.NORMALISATION_MEANVAR)
        The filter normalisation policy. See also :class:`FilterGenerator`

    poolings : iterable of triple (height, width, policy) (default :
    [(3, 3, Const.POOLING_AGGREG_AVG)])
        A list of parameters to instanciate the according :class:`Pooler`
        height : int > 0
            the height of the neighborhood window
        width : int > 0
            the width of the neighborhood window
        policy : int in {Const.POOLING_NONE, Const.POOLING_AGGREG_MIN,
    Const.POOLING_AGGREG_AVG, Const.POOLING_AGGREG_MAX,
    Const.POOLING_CONV_MIN, Const.POOLING_CONV_AVG, Const.POOLING_CONV_MAX}

    nbSubwindows : int >= 0 (default : 10)
        The number of subwindow to extract
    subwindowMinSizeRatio : float > 0 (default : 0.5)
        The minimum size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowMaxSizeRatio : float : subwindowMinSizeRatio
    <= subwindowMaxSizeRatio <= 1 (default : 1.)
        The maximim size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowTargetWidth : int > 0 (default : 16)
        The width of the subwindows after reinterpolation
    subwindowTargetHeight : int > 0 (default : 16)
        The height of the subwindows after reinterpolation
    subwindowInterpolation : int (default :
    SubWindowExtractor.INTERPOLATION_BILINEAR)
        The subwindow reinterpolation algorithm. For more information, see
        :class:`SubWindowExtractor`

    includeOriginalImage : boolean (default : False)
        Whether or not to include the original image in the subwindow
        extraction process

    nbJobs : int >0 or -1 (default : -1)
        The number of process to spawn for parallelizing the computation.
        If -1, the maximum number is selected. See also :mod:`Joblib`.
    verbosity : int >= 0 (default : 10)
        The verbosity level
    tempFolder : string (directory path) (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :class:`ParallelCoordinator`)

    random : bool (default : True)
        Whether to use randomness or use a predefined seed

    Return
    ------
        coordinator : :class:`Coordinator`
            The RandConvCoordinator corresponding to the
            set of parameters

    Notes
    -----
    - Filter generator
        Base instance of :class:`Finite3SameFilter` with a base instance of
        :class:`NumberGenerator` for the values and
        :class:`OddUniformGenerator` for the sizes
    - Filter size
        The filter are square (same width as height)
    - Convolver
        Base instance of :class:`RGBConvolver`
    - Subwindow random generator
        The subwindow random generator is a :class:`NumberGenerator` base
        instance (generate real nubers uniformely).
    - Feature extractor
        Base instance of :class:`ImageLinearizationExtractor`
    """
    #RANDOMNESS
    swngSeed = 0
    filtValGenSeed = 1
    filtSizeGenSeed = 2
    if random is None:
        swngSeed = None
        filtValGenSeed = None
        filtSizeGenSeed = None

    #CONVOLUTIONAL EXTRACTOR
    #Filter generator
    filterValGenerator = NumberGenerator(filterMinVal, filterMaxVal,
                                         seed=filtValGenSeed)
    filterSizeGenerator = OddUniformGenerator(filterMinSize, filterMaxSize,
                                              seed=filtSizeGenSeed)
    baseFilterGenerator = FilterGenerator(filterValGenerator,
                                          filterSizeGenerator,
                                          normalisation=filterNormalisation)
    filterGenerator = Finite3SameFilter(baseFilterGenerator, nbFilters)

    #Convolver
    convolver = RGBConvolver()

    #Aggregator
    poolers = []
    for height, width, policy in poolings:
        if policy == Const.POOLING_NONE:
            poolers.append(IdentityPooler())
        elif policy == Const.POOLING_AGGREG_AVG:
            poolers.append(AverageAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_AGGREG_MAX:
            poolers.append(MaximumAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_AGGREG_MIN:
            poolers.append(MinimumAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_CONV_MIN:
            poolers.append(ConvMinPooler(height, width))
        elif policy == Const.POOLING_CONV_AVG:
            poolers.append(ConvAvgPooler(height, width))
        elif policy == Const.POOLING_CONV_MAX:
            poolers.append(ConvMaxPooler(height, width))

    multiPooler = MultiPooler(poolers)

    #SubWindowExtractor
    swNumGenerator = NumberGenerator(seed=swngSeed)
    swExtractor = SubWindowExtractor(subwindowMinSizeRatio,
                                     subwindowMaxSizeRatio,
                                     subwindowTargetWidth,
                                     subwindowTargetHeight,
                                     subwindowInterpolation, swNumGenerator)

    multiSWExtractor = MultiSWExtractor(swExtractor, nbSubwindows, False)

    #ConvolutionalExtractor
    convolutionalExtractor = ConvolutionalExtractor(filterGenerator,
                                                    convolver,
                                                    multiSWExtractor,
                                                    multiPooler,
                                                    includeOriginalImage)
    #FEATURE EXTRACTOR
    featureExtractor = ImageLinearizationExtractor()

    #LOGGER
    autoFlush = verbosity >= 45
    logger = ProgressLogger(StandardLogger(autoFlush=autoFlush,
                                           verbosity=verbosity))
    #COORDINATOR
    coordinator = RandConvCoordinator(convolutionalExtractor, featureExtractor,
                                      logger, verbosity)

    if nbJobs != 1:
        coordinator.parallelize(nbJobs, tempFolder)
    return coordinator


def customRandConvFactory(
        poolings=[(3, 3, Const.POOLING_AGGREG_AVG)],
        nbSubwindows=10,
        subwindowMinSizeRatio=0.5, subwindowMaxSizeRatio=1.,
        subwindowTargetWidth=16, subwindowTargetHeight=16,
        subwindowInterpolation=SubWindowExtractor.INTERPOLATION_BILINEAR,
        includeOriginalImage=False,
        nbJobs=-1, verbosity=10, tempFolder=None,
        random=True):
    """
    Factory method to create :class:`RandConvCoordinator` tuned for RGB images
    using predefined well-known filters

    Parameters
    ----------
    poolings : iterable of triple (height, width, policy) (default :
    [(3, 3, Const.POOLING_AGGREG_AVG)])
        A list of parameters to instanciate the according :class:`Pooler`
        height : int > 0
            the height of the neighborhood window
        width : int > 0
            the width of the neighborhood window
        policy : int in {Const.POOLING_NONE, Const.POOLING_AGGREG_MIN,
    Const.POOLING_AGGREG_AVG, Const.POOLING_AGGREG_MAX,
    Const.POOLING_CONV_MIN, Const.POOLING_CONV_AVG, Const.POOLING_CONV_MAX}

    nbSubwindows : int >= 0 (default : 10)
        The number of subwindow to extract
    subwindowMinSizeRatio : float > 0 (default : 0.5)
        The minimum size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowMaxSizeRatio : float : subwindowMinSizeRatio
    <= subwindowMaxSizeRatio <= 1 (default : 1.)
        The maximim size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowTargetWidth : int > 0 (default : 16)
        The width of the subwindows after reinterpolation
    subwindowTargetHeight : int > 0 (default : 16)
        The height of the subwindows after reinterpolation
    subwindowInterpolation : int (default :
    SubWindowExtractor.INTERPOLATION_BILINEAR)
        The subwindow reinterpolation algorithm. For more information, see
        :class:`SubWindowExtractor`

    includeOriginalImage : boolean (default : False)
        Whether or not to include the original image in the subwindow
        extraction process

    nbJobs : int >0 or -1 (default : -1)
        The number of process to spawn for parallelizing the computation.
        If -1, the maximum number is selected. See also :mod:`Joblib`.
    verbosity : int >= 0 (default : 10)
        The verbosity level
    tempFolder : string (directory path) (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :class:`ParallelCoordinator`)

    random : bool (default : True)
        Whether to use randomness or use a predefined seed

    Return
    ------
        coordinator : :class:`Coordinator`
            The RandConvCoordinator corresponding to the
            set of parameters

    Notes
    -----
    - Convolver
        Base instance of :class:`RGBConvolver`
    - Subwindow random generator
        The subwindow random generator is a :class:`NumberGenerator` base
        instance (generate real nubers uniformly).
    - Feature extractor
        Base instance of :class:`ImageLinearizationExtractor`
    """
    #RANDOMNESS
    swngSeed = 0
    if random is None:
        swngSeed = None

    #CONVOLUTIONAL EXTRACTOR
    filterGenerator = customFinite3sameFilter()

    #Convolver
    convolver = RGBConvolver()

    #Aggregator
    poolers = []
    for height, width, policy in poolings:
        if policy == Const.POOLING_NONE:
            poolers.append(IdentityPooler())
        elif policy == Const.POOLING_AGGREG_AVG:
            poolers.append(AverageAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_AGGREG_MAX:
            poolers.append(MaximumAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_AGGREG_MIN:
            poolers.append(MinimumAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_CONV_MIN:
            poolers.append(ConvMinPooler(height, width))
        elif policy == Const.POOLING_CONV_AVG:
            poolers.append(ConvAvgPooler(height, width))
        elif policy == Const.POOLING_CONV_MAX:
            poolers.append(ConvMaxPooler(height, width))

    multiPooler = MultiPooler(poolers)

    #SubWindowExtractor
    swNumGenerator = NumberGenerator(seed=swngSeed)
    swExtractor = SubWindowExtractor(subwindowMinSizeRatio,
                                     subwindowMaxSizeRatio,
                                     subwindowTargetWidth,
                                     subwindowTargetHeight,
                                     subwindowInterpolation, swNumGenerator)

    multiSWExtractor = MultiSWExtractor(swExtractor, nbSubwindows, False)

    #ConvolutionalExtractor
    convolutionalExtractor = ConvolutionalExtractor(filterGenerator,
                                                    convolver,
                                                    multiSWExtractor,
                                                    multiPooler,
                                                    includeOriginalImage)
    #FEATURE EXTRACTOR
    featureExtractor = ImageLinearizationExtractor()

    #LOGGER
    autoFlush = verbosity >= 45
    logger = ProgressLogger(StandardLogger(autoFlush=autoFlush,
                                           verbosity=verbosity))
    #COORDINATOR
    coordinator = RandConvCoordinator(convolutionalExtractor, featureExtractor,
                                      logger, verbosity)

    if nbJobs != 1:
        coordinator.parallelize(nbJobs, tempFolder)
    return coordinator


def coordinatorCompressRandConvFactory(
        nbFilters=5,
        filterMinVal=-1, filterMaxVal=1,
        filterMinSize=1, filterMaxSize=17,
        filterNormalisation=FilterGenerator.NORMALISATION_MEANVAR,
        poolings=[(3, 3, Const.POOLING_AGGREG_AVG)],
        nbSubwindows=10,
        subwindowMinSizeRatio=0.5, subwindowMaxSizeRatio=1.,
        subwindowTargetWidth=16, subwindowTargetHeight=16,
        subwindowInterpolation=SubWindowExtractor.INTERPOLATION_BILINEAR,
        includeOriginalImage=False,
        compressorType="Sampling", nbCompressedFeatures=1,
        compressOriginalImage=True,
        nbJobs=-1, verbosity=10, tempFolder=None,
        random=True):
    """
    Factory method to create :class:`RandConvCoordinator` tuned for RGB images

    Parameters
    ----------
    nbFilters : int >= 0 (default : 5)
        The number of filter
    filterMinVal : float (default : -1)
        The minimum value of a filter component
    filterMaxVal : float : filterMinVal <= filterMaxVal (default : 1)
        The maximum value of a filter component
    filterMinSize : int >= 0 : odd number (default : 1)
        The minimum size of a filter
    filterMaxSize : int >= 0 : odd number s.t.  filterMinSize <= filterMaxSize
    (default : 1)
        The maximum size of a filter
    filterNormalisation : int (default : FilterGenerator.NORMALISATION_MEANVAR)
        The filter normalisation policy. See also :class:`FilterGenerator`

    poolings : iterable of triple (height, width, policy) (default :
    [(3, 3, Const.POOLING_AGGREG_AVG)])
        A list of parameters to instanciate the according :class:`Pooler`
        height : int > 0
            the height of the neighborhood window
        width : int > 0
            the width of the neighborhood window
        policy : int in {Const.POOLING_NONE, Const.POOLING_AGGREG_MIN,
    Const.POOLING_AGGREG_AVG, Const.POOLING_AGGREG_MAX,
    Const.POOLING_CONV_MIN, Const.POOLING_CONV_AVG, Const.POOLING_CONV_MAX}

    nbSubwindows : int >= 0 (default : 10)
        The number of subwindow to extract
    subwindowMinSizeRatio : float > 0 (default : 0.5)
        The minimum size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowMaxSizeRatio : float : subwindowMinSizeRatio
    <= subwindowMaxSizeRatio <= 1 (default : 1.)
        The maximim size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowTargetWidth : int > 0 (default : 16)
        The width of the subwindows after reinterpolation
    subwindowTargetHeight : int > 0 (default : 16)
        The height of the subwindows after reinterpolation
    subwindowInterpolation : int (default :
    SubWindowExtractor.INTERPOLATION_BILINEAR)
        The subwindow reinterpolation algorithm. For more information, see
        :class:`SubWindowExtractor`

    includeOriginalImage : boolean (default : False)
        Whether or not to include the original image in the subwindow
        extraction process

    compressorType : str (default : "Sampling")
        The type of compressor to use. One of the {"Sampling", "PCA"}
    nbCompressedFeatures : int > 0 (default : 1)
        The number features per filter after compression. It should be
        inferior to the number of features per filter traditionally output by
        the :class:`randConvCoordinator`
    compressOriginalImage : boolean (default : True)
        Whether to compress the original (if included) as well as the other
        features

    nbJobs : int >0 or -1 (default : -1)
        The number of process to spawn for parallelizing the computation.
        If -1, the maximum number is selected. See also :mod:`Joblib`.
    verbosity : int >= 0 (default : 10)
        The verbosity level
    tempFolder : string (directory path) (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :class:`ParallelCoordinator`)

    random : bool (default : True)
        Whether to use randomness or use a predefined seed
    Return
    ------
        coordinator : :class:`Coordinator`
            The RandConvCoordinator (possibly decorated) corresponding to the
            set of parameters

    Notes
    -----
    - Filter generator
        Base instance of :class:`Finite3SameFilter` with a base instance of
        :class:`NumberGenerator` for the values and
        :class:`OddUniformGenerator` for the sizes
    - Filter size
        The filter are square (same width as height)
    - Convolver
        Base instance of :class:`RGBConvolver`
    - Aggregator
        Base instance of :class:`AverageAggregator`
    - Subwindow random generator
        The subwindow random generator is a :class:`NumberGenerator` base
        instance (generate real nubers uniformely).
    - Feature extractor
        Base instance of :class:`ImageLinearizationExtractor`
    """

    #RANDOMNESS
    swngSeed = 0
    filtValGenSeed = 1
    filtSizeGenSeed = 2
    if random is None:
        swngSeed = None
        filtValGenSeed = None
        filtSizeGenSeed = None

    #CONVOLUTIONAL EXTRACTOR
    #Filter generator
    filterValGenerator = NumberGenerator(filterMinVal, filterMaxVal,
                                         seed=filtValGenSeed)
    filterSizeGenerator = OddUniformGenerator(filterMinSize, filterMaxSize,
                                              seed=filtSizeGenSeed)
    baseFilterGenerator = FilterGenerator(filterValGenerator,
                                          filterSizeGenerator,
                                          normalisation=filterNormalisation)
    filterGenerator = Finite3SameFilter(baseFilterGenerator, nbFilters)

    #Convolver
    convolver = RGBConvolver()

    #Aggregator
    poolers = []
    for height, width, policy in poolings:
        if policy == Const.POOLING_NONE:
            poolers.append(IdentityPooler())
        elif policy == Const.POOLING_AGGREG_AVG:
            poolers.append(AverageAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_AGGREG_MAX:
            poolers.append(MaximumAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_AGGREG_MIN:
            poolers.append(MinimumAggregator(width, height,
                                             subwindowTargetWidth,
                                             subwindowTargetHeight))
        elif policy == Const.POOLING_CONV_MIN:
            poolers.append(ConvMinPooler(height, width))
        elif policy == Const.POOLING_CONV_AVG:
            poolers.append(ConvAvgPooler(height, width))
        elif policy == Const.POOLING_CONV_MAX:
            poolers.append(ConvMaxPooler(height, width))

    multiPooler = MultiPooler(poolers)

    #SubWindowExtractor
    swNumGenerator = NumberGenerator(seed=swngSeed)
    swExtractor = SubWindowExtractor(subwindowMinSizeRatio,
                                     subwindowMaxSizeRatio,
                                     subwindowTargetWidth,
                                     subwindowTargetHeight,
                                     subwindowInterpolation, swNumGenerator)

    multiSWExtractor = MultiSWExtractor(swExtractor, nbSubwindows, False)

    #ConvolutionalExtractor
    convolutionalExtractor = ConvolutionalExtractor(filterGenerator,
                                                    convolver,
                                                    multiSWExtractor,
                                                    multiPooler,
                                                    includeOriginalImage)
    #FEATURE EXTRACTOR
    featureExtractor = ImageLinearizationExtractor()

    #LOGGER
    autoFlush = verbosity >= 45
    logger = ProgressLogger(StandardLogger(autoFlush=autoFlush,
                                           verbosity=verbosity))

    #COMPRESSOR
    if compressorType == "PCA":
        compressorFactory = PCAFactory(nbCompressedFeatures)
    else:
        #"Sampling"
        compressorFactory = SamplerFactory(nbCompressedFeatures)

    #COORDINATOR
    coordinator = CompressRandConvCoordinator(convolutionalExtractor,
                                              featureExtractor,
                                              compressorFactory,
                                              compressOriginalImage,
                                              logger, verbosity)

    if nbJobs != 1:
        coordinator.parallelize(nbJobs, tempFolder)
    return coordinator


if __name__ == "__main__":

    from CifarLoader import CifarLoader
    from ImageBuffer import ImageBuffer

    path = "data_batch_1"
    imgBuffer = CifarLoader(path, outputFormat=ImageBuffer.NUMPY_FORMAT)

    #coord = coordinatorRandConvFactory(verbosity=50)
    coord = coordinatorRandConvFactory(nbJobs=1, verbosity=0)

    X, y = coord.process(imgBuffer[0:10])
    print X.shape, len(y)
